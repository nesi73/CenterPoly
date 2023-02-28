'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

import json 
import cv2
import numpy as np
import torch.utils.data as data
import os

OUTPUT_PATH = './src/lib/datasets/copilot'

def order_four_points(four_points):
    """
    With the four corners of the panel given, we select the two leftmost with respect to the x coordinates and from these 
    we select the one that is higher, making this reference to point zero. For the other points, the same logic is followed.
    @four_points: corners of the panel
    Organization of the corners:
        p0---------p1
        |          |
        |          |
        p2---------p3
    """

    four_points=[[point[0][0], point[0][1]] for point in four_points]
    four_points=np.array(four_points)
    sort_x_left = four_points[four_points[:, 0].argsort()][:2]
    sort_x_right = four_points[four_points[:, 0].argsort()][2:]
    sort_y_left = sort_x_left[sort_x_left[:, 1].argsort()]
    sort_y_right = sort_x_right[sort_x_right[:, 1].argsort()]

    points = [sort_y_left[0][0], sort_y_left[0][1]], [sort_y_right[0][0] , sort_y_right[0][1] ], [sort_y_left[1][0], sort_y_left[1][1] ], [sort_y_right[1][0], sort_y_right[1][1]]

    final_pts=[]
    for pt in points:
        for i, fp in enumerate(four_points):
            if pt[0] == fp[0] and pt[1] == fp[1]:
                final_pts.append(four_points[i].tolist())
    
    return final_pts

def _bbox_to_coco_bbox(bbox):
    return [(bbox[0][0]), (bbox[0][1]),
            (bbox[1][0] - bbox[0][0]), (bbox[2][1] - bbox[0][1])]

def polygon_to_bbox(cnt):
    alpha=0.01
    epsilon = alpha*cv2.arcLength(cnt,True)
    box = cv2.approxPolyDP(cnt,epsilon,True)
    box=np.int0(box)

    while len(box) > 4 and epsilon < 7:
        alpha=alpha+0.01
        epsilon = alpha*cv2.arcLength(cnt,True)
        box = cv2.approxPolyDP(cnt,epsilon,True)
        box=np.int0(box)

    if len(box) == 4:
        return box.tolist()
    else:
        return []

if __name__ == '__main__':

  cats = ['panel', 'cropped_panel']
  cat_ids = {cat: i + 1 for i, cat in enumerate(cats)} #TODO: add i + 1

  cat_info = []
  for i, cat in enumerate(cats):
      cat_info.append({'name': cat, 'id': i + 1})

  ret_train = {'images': [], 'annotations': [], "categories": cat_info}
  ret_val = {'images': [], 'annotations': [], "categories": cat_info}
  ret_test = {'images': [], 'annotations': [], "categories": cat_info}

  with open('./src/lib/datasets/labelbox.json', 'r') as f:
      labelbox_data = json.load(f)


  for i, data in enumerate(labelbox_data):
      image_id = data['ID']
      image_info = {'file_name': data['External ID'],
                      'id': image_id}        
      
      if i < 0.8*len(labelbox_data):
          ret_train['images'].append(image_info)
      elif i < 0.9*len(labelbox_data):
          ret_val['images'].append(image_info)
      else:
          ret_test['images'].append(image_info)

      for label in data['Label']['objects']:
          cat_id=cat_ids[label['value']]
          polygon = [[point['x'], point['y']] for point in label['polygon']]
          
          bbox=polygon_to_bbox(np.array(polygon).astype(np.int32))

          if len(bbox) == 0:
              continue
          
          bbox=order_four_points(bbox)
          occluded=cat_id #fully visible if is a panel, occluded if is a cropped panel
          truncated=cat_id

          if i < 0.8*len(labelbox_data):
            id = int(len(ret_train['annotations']) + 1)
          elif i < 0.9*len(labelbox_data):
            id = int(len(ret_val['annotations']) + 1)
          else:
            id = int(len(ret_test['annotations']) + 1)

          ann = {'image_id': image_id,
                  'id': id,
                  'category_id': cat_id,
                  'segmentation': [polygon],
                  'bbox': _bbox_to_coco_bbox(bbox),
                  'truncated': truncated,
                  'occluded': occluded}
          
          if i < 0.8*len(labelbox_data):
            ret_train['annotations'].append(ann)
          elif i < 0.9*len(labelbox_data):
            ret_val['annotations'].append(ann)
          else:
            ret_test['annotations'].append(ann)

  out_path = '{}/annotations/instances_train2017.json'.format(OUTPUT_PATH)
  json.dump(ret_train, open(out_path, 'w'))

  out_path = '{}/annotations/instances_val2017.json'.format(OUTPUT_PATH)
  json.dump(ret_val, open(out_path, 'w'))

  out_path = '{}/annotations/instances_test2017.json'.format(OUTPUT_PATH)
  json.dump(ret_test, open(out_path, 'w'))