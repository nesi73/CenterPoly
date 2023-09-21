import cv2
from matplotlib import pyplot as plt
import json
import os
import numpy as np

def paint_bbox(all_bbox, im,seg):
    for i,bbox in enumerate(all_bbox):
        bbox=[int(float(bbox[0])),int(float(bbox[1])),int(float(bbox[2])),int(float(bbox[3]))]
        new_seg=np.array(seg[i][0])
        new_seg=new_seg.astype(int)

        # cv2.circle(im, (bbox[0], bbox[1]), 10, (255,0,0))
        #cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2]+bbox[0], bbox[3]+bbox[1]), color=(255,0,0))
        new_seg=np.array(new_seg).reshape(-1,1,2)
        cv2.drawContours(im,[new_seg], -1,(255,0,0),3)
        bbox=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1],  bbox[0], bbox[1]+bbox[3] ]
        box=np.array(bbox).reshape(-1,1,2)
        cv2.drawContours(im,[box], -1,(255,255,0),4)
    #plt.imshow(im)
    #plt.show()
    cv2.imwrite('test.jpg', im)

if __name__ == '__main__':

    with open('/home/cvar_user/Desktop/CenterPoly/src/lib/datasets/copilot_alcala_resized/annotations/instances_train2017.json', 'r') as f:
        data=json.load(f)
        filename=data['images'][0]['file_name']
        id_image=data['images'][0]['id']

        img=cv2.imread('/home/cvar_user/Desktop/CenterPoly/src/lib/datasets/copilot_alcala_resized/images/train2017/'+filename)
        
        all_bbox=[]
        all_seg=[]
        for annotation in data['annotations']:
            if annotation['image_id']!=id_image:
                continue
            all_bbox.append(annotation['bbox'])
            all_seg.append(annotation['segmentation'])
        paint_bbox(all_bbox, img, all_seg)