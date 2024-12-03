""" from models.yolo import Model
import torch

def intersect_dicts(da, db, exclude=()):
    #Returns intersection of `da` and `db` dicts with matching keys and shapes, excluding `exclude` keys; uses `da` values.
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

device="cuda" if torch.cuda.is_available() else "cpu"
weights='yolov5s.pt'
ckpt = torch.load(weights, map_location=device)
model = Model(cfg='models/yolov5s.yaml',ch=3,nc=80).to(device)  # create
csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
model.load_state_dict(csd,strict=False)  # load
for k, v in model.named_parameters():
    v.requires_grad = False  # freeze all layers
model.eval()
 """
import torch
import os
device="cuda" if torch.cuda.is_available() else "cpu"
filePath='/aiarena/gpfs'
checkpoints=[]
for i in os.listdir(filePath):
    if i.startswith('checkpoint'):
        checkpoints.append(filePath+'/'+i)
for checkpoint in checkpoints:
    ckpt=torch.load(checkpoint,map_location=device)
    csd=ckpt['state_dict']
    state={'state_dict':csd}
    torch.save(state,checkpoint[:-8]+'.pt')
    os.remove(checkpoint)
