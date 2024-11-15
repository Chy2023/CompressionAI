import json
import os
import numpy as np
path='/aiarena/group/icgroup/data/coco/annotations/instances_val2017.json'
root='/aiarena/gpfs/labels'
val=json.load(open(path,'r'))
annos=val['annotations']
images=val['images']
categories=val['categories']
categories.sort(key=lambda x:x['id'])
ids=[i['id'] for i in categories]
index_inverse=dict(enumerate(ids))
index={j:i for i,j in index_inverse.items()}
#create mapping between ids

img_path='/aiarena/group/icgroup/data/coco/val2017'
img_list=os.listdir(img_path)
img_list=[os.path.splitext(i)[0] for i in img_list]
for i in img_list:
    os.mknod(root+'/'+i+'.txt')
#create folder and files for images,called only once

#add annotations
for anno in annos:
    category_id=index[anno['category_id']]
    bbox=anno['bbox']
    bbox.insert(0,category_id)
    bbox[1]=bbox[1]+0.5*bbox[3]
    bbox[2]=bbox[2]+0.5*bbox[4]
    image_id=str(anno['image_id'])
    image_id='0'*(12-len(image_id))+image_id
    pathname=root+'/'+image_id+'.txt'
    with open(root+'/'+image_id+'.txt','a') as f:
        for i in bbox:
            f.write(str(i)+' ')
        f.write('\n')

#scaled to (0,1)
for image in images:
    file_name=image['file_name']
    h,w=image['height'],image['width']
    file_data=''
    pathname=root+'/'+os.path.splitext(file_name)[0]+'.txt'
    with open(pathname,'r') as f:
        while True:
            line=f.readline()
            if not line:
                break
            line=line.split(' ')
            line[1]=str(round(float(line[1])/w,6))
            line[3]=str(round(float(line[3])/w,6))
            line[2]=str(round(float(line[2])/h,6))
            line[4]=str(round(float(line[4])/h,6))
            line=' '.join(line)
            file_data+=line
    with open(pathname,'w') as f:
        f.write(file_data)