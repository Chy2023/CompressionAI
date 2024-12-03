#install packages
pip install imgaug
pip install opencv-python-headless
pip install terminaltables
pip install -r requirements.txt
#VPN：
: '
export http_proxy=192.168.163.189:7890;export https_proxy=192.168.163.189:7890

unset http_proxy;unset https_proxy
'
#train
: '
python _train.py -d /aiarena/group/icgroup/data/flicker -e 200 -q X --checkpoint checkpoint_XXX.pth.tar --id XX --gamma X
'
#copy font file for _test.py
: '
python << EOF
from ultralytics import YOLO
import os
os.system('cp utils/Arial.ttf /root/.config/Ultralytics')
EOF
'
#copy pretrained models for _baselines.py
: '
python << EOF
import os
src='/aiarena/gpfs/baseline_models'
model_list=os.listdir(src)
model_list=[src+'/'+i for i in model_list]
dest='/root/.cache/torch/hub/checkpoints'
os.makedirs(dest)
for i in model_list:
    os.system(f'cp {i} {dest}')
EOF
'
#test
: '
python _test.py --img 640  --data data/coco.yaml  --checkpoint checkpoint_XXX.pth.tar
'