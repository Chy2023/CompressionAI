import torch
import torch.nn.functional as F
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
    cheng2020_attn
)
import argparse
import math
from pytorch_msssim import ms_ssim
from model import Choi2022
import os
from torchvision import transforms
from PIL import Image



device='cuda' if torch.cuda.is_available() else 'cpu'
model_list=[
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
    cheng2020_attn
]
p=64


#load pretrained compress network
def load_network(model_type,quality):
    model=model_type(quality=quality,pretrained=True).eval().to(device)
    return model

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def evaluate(path,img_list,model):
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
        x = img.unsqueeze(0)
        x_padded, padding = pad(x, p)
        count += 1
        with torch.no_grad():
            out_net = model.forward(x_padded)
            out_net['x_hat'].clamp_(0, 1)
            Bit_rate += compute_bpp(out_net)
            out_net["x_hat"] = crop(out_net["x_hat"], padding)
            PSNR += compute_psnr(x, out_net["x_hat"])
            MS_SSIM += compute_msssim(x, out_net["x_hat"])
    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    print(f'average_Bit-rate: {Bit_rate:.2f} bpp')
    print(f'average_PSNR: {PSNR:.2f} dB')
    print(f'average_MS-SSIM: {MS_SSIM:.2f} dB')
    
    
def run():
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str,default="/aiarena/group/icgroup/data/kodak", help="Path to dataset")
    args = parser.parse_args()
    path=args.data
    img_list=[file for file in os.listdir(path)]
    if args.checkpoint is not None:
        model=Choi2022()
        model=model.to(device)
        print("Loading", args.checkpoint)
        checkpoint=torch.load(args.checkpoint, map_location=device)
        _dict={}
        for k, v in checkpoint["state_dict"].items():
                _dict[k.replace("module.", "")] = v
        model.load_state_dict(_dict)
        model.eval()
        evaluate(path,img_list,model)
    else:
        for model_type in model_list:
            for i in range(1,7):
                print(model_type,i)
                model=load_network(model_type,i)
                evaluate(path,img_list,model)
            
if __name__ == "__main__":
    run()