import os
dest='/aiarena/gpfs/baseline_models'
src='/root/.cache/torch/hub/checkpoints'
model_list=os.listdir(src)
model_list=[src+'/'+i for i in model_list]
for i in model_list:
    os.system(f'cp {i} {dest}')