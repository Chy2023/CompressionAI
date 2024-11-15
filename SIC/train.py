import argparse
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms


from pytorch_msssim import ms_ssim
from compressai.optimizers import net_aux_optimizer
from model import Choi2022
import math

from typing import Any, Dict, Mapping, cast
from compressai.registry import OPTIMIZERS

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import os
import datetime
import pytz
import wandb
os.environ["WANDB_API_KEY"]='0605d04eb3ad284166eaf3e324a3fd9c51a8559b'
lmbda_list=[0,0.0018,0.0035,0.0067,0.013,0.025,0.0483]
run_name=datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H:%M:%S")

class ImageFolder(Dataset):

    def __init__(self, root,size, transform=None, split="train"):
        

        #self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        self.samples=[]
        if root.endswith(".txt"):
            with open(root, "r") as file:
                img_files = file.readlines()
            img_files=[line.rstrip() for line in img_files]
            for f in img_files:
                if not os.path.isfile(f):
                    continue
                img=Image.open(f)
                if img.width<size or img.height<size:
                    continue 
                self.samples.append(f)
        else:
            splitdir = Path(root) / split
            if not splitdir.is_dir():
                raise RuntimeError(f'Missing directory "{splitdir}"')
            for f in splitdir.iterdir():
                if not f.is_file():
                    continue
                img=Image.open(f)
                if img.width<size or img.height<size:
                    continue 
                self.samples.append(f)
        self.samples=sorted(self.samples)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)




def net_aux_optimizer(
    net: nn.Module, conf: Mapping[str, Any]
) -> Dict[str, optim.Optimizer]:
    """Returns separate optimizers for net and auxiliary losses.

    Each optimizer operates on a mutually exclusive set of parameters.
    """
    parameters = {
        "net": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        },
        "aux": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        },
    }

    # Make sure we don't have an intersection of parameters
    #params_dict = dict(net.named_parameters())
    params_dict={name:param for name,param in net.named_parameters() if param.requires_grad}
    inter_params = parameters["net"] & parameters["aux"]
    union_params = parameters["net"] | parameters["aux"]
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    def make_optimizer(key):
        kwargs = dict(conf[key])
        del kwargs["type"]
        params = (params_dict[name] for name in sorted(parameters[key]))
        return OPTIMIZERS[conf[key]["type"]](params, **kwargs)

    optimizer = {key: make_optimizer(key) for key in ["net", "aux"]}

    return cast(Dict[str, optim.Optimizer], optimizer)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, gamma=0.006, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.gamma=gamma
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out['feature_mse_loss']=nn.MSELoss()(output['feature'],output['feature_target'])
        distortion=self.gamma*(255**2)*out['feature_mse_loss']
        #distortion=self.gamma*out['feature_mse_loss']
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion += 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion += 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    model.yolo.eval()
    device = next(model.parameters()).device


    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)


        out_criterion = criterion(out_net, d)

        
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f'\tfeature_mse_loss: {out_criterion["feature_mse_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

#TODO:add yolo module in test stage
def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    feature_mse_loss=AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            feature_mse_loss.update(out_criterion["feature_mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tfeature_mse_loss: {feature_mse_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg,mse_loss.avg,bpp_loss.avg,feature_mse_loss.avg,aux_loss.avg



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q"
        "--quality-index",
        dest="quality",
        type=int,
        default=6,
        help="Bit-rate distortion parameter index,range from 1 to 6(default: %(default)s)",
    )
    parser.add_argument(
        "--gamma",
        dest="gamma",
        type=float,
        default=6e-3,
        help="task-relevant feature distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        #default=(256, 256),
        default=(416, 416),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--id", type=str, help="Id to resume a training")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    if args.id is not None:
        wandb.init(project='Choi2022',name=f'experiment_{run_name}',config=vars(args),resume=True,id=args.id)
    else:
        wandb.init(project='Choi2022',name=f'experiment_{run_name}',config=vars(args))
    
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )
    size=args.patch_size[0]
    train_dataset = ImageFolder(args.dataset, size,split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, size,split="test", transform=test_transforms)
    #train on coco_val2017 dataset
    """ train_dataset = ImageFolder(args.dataset, size,split="val2017", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, size,split="val2017", transform=test_transforms) """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net=Choi2022()
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    
    lmbda=lmbda_list[args.quality]
    criterion = RateDistortionLoss(lmbda=lmbda,gamma=args.gamma)

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        if torch.cuda.device_count()==1:
            _dict={}
            for k, v in checkpoint["state_dict"].items():
                    _dict[k.replace("module.", "")] = v
            net.load_state_dict(_dict)
        else:
            net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_loss=checkpoint['loss']

    
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss,mse_loss,bpp_loss,feature_mse_loss,aux_loss= test_epoch(epoch, test_dataloader, net, criterion)
        wandb.log({'lr':optimizer.param_groups[0]['lr'],'loss':loss,'bpp_loss':bpp_loss,'mse_loss':mse_loss,'feature_mse_loss':feature_mse_loss,'aux_loss':aux_loss})        
        lr_scheduler.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if is_best:
            state={
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
            torch.save(state,f'checkpoint_{run_name}.pth.tar')
    wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])
