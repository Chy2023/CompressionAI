#install packages
pip install imgaug
pip install opencv-python-headless
pip install terminaltables
#VPN：
#export http_proxy=192.168.163.144:7890
#export https_proxy=192.168.163.144:7890
#train
#python SIC/train.py -d /aiarena/group/icgroup/data/flicker -e 200 -q X --checkpoint checkpoint_XXX.pth.tar --id XX --gamma X
#test
#python SIC/test.py --checkpoint checkpoint_XXX.pth.tar
#eval
#python SIC/eval.py --checkpoint checkpoint_XXX.pth.tar