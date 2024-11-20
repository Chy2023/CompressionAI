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
import tqdm
import numpy as np
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
import math
from pytorch_msssim import ms_ssim

import os
import sys
sys.path.append(".")
sys.path.append("./YOLOv3")
from YOLOv3.pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression,  xywh2xyxy
from YOLOv3.pytorchyolo.utils.datasets import ListDataset
from YOLOv3.pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from YOLOv3.pytorchyolo.models import load_model


device='cuda' if torch.cuda.is_available() else 'cpu'
model_list=[
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
    cheng2020_attn
]

#pad 416x416 to 448x448
def pad(x):
    padding=(448-416)//2
    x=F.pad(x,(padding,padding,padding,padding))
    return x
#unpad 448x448 to 416x416
def crop(x):
    padding=(448-416)//2*(-1)
    x=F.pad(x,(padding,padding,padding,padding))
    return x
#load pretrained yolo
def load_yolo(config_path='YOLOv3/config/yolov3.cfg',weights_path='YOLOv3/yolov3.weights'):
    root=os.getcwd()
    config_path=root+'/'+config_path
    weights_path=root+'/'+weights_path
    model=load_model(config_path,weights_path)
    model.eval()
    return model
#load pretrained compress network
def load_network(model_type,quality):
    model=model_type(quality=quality,pretrained=True).eval().to(device)
    return model



def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()


def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


def _evaluate(model, yolo,dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    model.eval()  # Set model to evaluation mode
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    bpp=0
    count=0
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        imgs=imgs.to(device)
        imgs_padded=pad(imgs)
        with torch.no_grad():
            outputs = model(imgs_padded)
            bpp+=compute_bpp(outputs)
            outputs['x_hat']=crop(outputs['x_hat'])
            imgs=outputs['x_hat'].clamp(0,1)
            yolo_outputs=yolo(imgs)
            yolo_outputs = non_max_suppression(yolo_outputs, conf_thres=conf_thres, iou_thres=nms_thres)
        count+=1
        sample_metrics += get_batch_statistics(yolo_outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)
    bpp=bpp/count
    print_eval_stats(metrics_output, class_names, verbose)
    print(f'average_Bit-rate: {bpp:.2f} bpp')
    return metrics_output

def evaluate_model_file(model, yolo,img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)
    metrics_output = _evaluate(
        model,
        yolo,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output

def run():
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()

    valid_path='/aiarena/gpfs/YOLO/data/coco/5k.txt'
    class_names =load_classes('/aiarena/gpfs/YOLOv3/data/coco.names')  # List of class names
    yolo=load_yolo()
    for model_type in model_list:
        for i in range(1,7):
            if model_type !=bmshj2018_factorized or i!=5:
                continue
            print(model_type,i)
            model=load_network(model_type,i)
            precision, recall, AP, f1, ap_class = evaluate_model_file(
                model,
                yolo,
                valid_path,
                class_names,
                batch_size=args.batch_size,
                img_size=args.img_size,
                n_cpu=args.n_cpu,
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=True)
if __name__ == "__main__":
    run()