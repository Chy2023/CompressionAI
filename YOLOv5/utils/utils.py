#some useful functions
import numpy as np
import torch
import math
from utils.general import xyxy2xywh, box_iou
from _model import Choi2022
from models.yolo import CustomModel

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

#--------------------------------------------------------------CUSTOM----------------------------------------------------------
def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    likelihoods=[out_net['likelihoods']['y_0'],out_net['likelihoods']['z']]
    return sum(torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
              for likelihood in likelihoods).item()


def load_model(checkpoint):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=Choi2022()
    model=model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def load_yolo(weights='yolov5s.pt',cfg='models/yolov5s.yaml'):
    device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(weights, map_location=device)
    model = CustomModel(cfg=cfg,ch=3,nc=80,anchors=None,cut_layer=3).to(device)  # create
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    model.load_state_dict(csd)  # load
    model.eval()
    return model

#--------------------------------------------------------------END-------------------------------------------------------------