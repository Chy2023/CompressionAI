#install packages
pip install imgaug
pip install opencv-python-headless
pip install terminaltables
#train
#python SIC/train.py -d /aiarena/group/icgroup/data/flicker -e 400 -q X --checkpoint checkpoint_XXX.pth.tar --id XX --gamma X
#test
#python SIC/test.py --checkpoint checkpoint_XXX.pth.tar
#eval
#python SIC/eval.py --checkpoint checkpoint_XXX.pth.tar