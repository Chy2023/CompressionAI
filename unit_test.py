#test annotations format
""" import numpy as np
l=np.loadtxt('/aiarena/gpfs/labels/000000000802.txt').reshape(-1,5)
print(l) """

#test .pth.tar file
""" import torch
device="cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path='checkpoint_best_loss.pth.tar'
checkpoint=torch.load(checkpoint_path,map_location=device)
last_epoch = checkpoint["epoch"] + 1
net=checkpoint["state_dict"]
for param in net:
    print(param)
optimizer=checkpoint["optimizer"]
aux_optimizer=checkpoint["aux_optimizer"]
lr_scheduler=checkpoint["lr_scheduler"]
best_loss=checkpoint['loss'] """

import torch
import torch.nn as nn
input = torch.ones(3, 5, requires_grad=True)
target = torch.zeros(3, 5)
result=nn.MSELoss()(input,target)
print(input,target,result)
