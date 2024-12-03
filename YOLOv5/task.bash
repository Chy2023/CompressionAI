python << EOF
import os
ckpts=['checkpoint_2024-11-26_20:55:53.pt',
'checkpoint_2024-11-27_23:28:00.pt',
'checkpoint_2024-11-27_23:28:20.pt',
'checkpoint_2024-11-26_20:25:14.pt',
'checkpoint_2024-11-27_23:28:39.pt',
'checkpoint_2024-11-26_20:54:58.pt']
#test
i=1
for ckpt in ckpts:
    print(f"{ckpt},q={i}")
    i+=1
    os.system(f'python _test.py --img 640  --data data/coco.yaml  --checkpoint {ckpt}')
#eval
i=1
for ckpt in ckpts:
    print(f"{ckpt},q={i}")
    i+=1
    os.system(f'python _eval.py --checkpoint {ckpt}')
#baselines
os.system(f'python _baselines.py --img 640  --data data/coco.yaml')
EOF

