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
from pycocotools.cocoeval import COCOeval
import os
import pycocotools.coco as coco

class COPILOT(data.Dataset):
    num_classes = 2
    OUTPUT_PATH = './src/lib/datasets/dataset'
    default_resolution=[1024,1024]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)

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

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        # coco_eval.params.catIds = [2, 3, 4, 6, 7, 8, 10, 11, 12, 13]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    

    def save_results(self, results, save_dir):
        print('-------------------------')
        print(self.convert_eval_format(results))
        print('-------------------------')

        json.dump(self.convert_eval_format(results), 
            open('{}/results.json'.format(save_dir), 'w'))

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

    def _to_float(self, x):
        return float("{:.2f}".format(x))
    
    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections
    
    def __len__(self):
        return self.num_samples

    def __init__(self, opt, split):
        super(COPILOT, self).__init__()

        self.data_dir = '/store/datasets/coco' if os.path.exists('/store/datasets/coco') else '/home/cvar_user/Desktop/CenterPoly/src/lib/datasets/copilot'
        print(self.data_dir)
        self.img_dir = os.path.join(self.data_dir, 'images',  '{}2017'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
            self.data_dir, 'annotations', 
            'image_info_test-dev2017.json').format(split)
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                self.data_dir, 'annotations', 
                'instances_extreme_{}2017.json').format(split)
            else:
                if split == 'val':
                    self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_{}2017.json').format(split)
                    # 'instances_{}2017_1_on_10.json').format(split)
                else:
                    self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_{}2017.json').format(split)

        self._valid_ids = [1, 2]
        self.cat_ids = {cat: i + 1 for i, cat in enumerate(self._valid_ids)} #TODO: add i + 1

        cat_info = []
        for i, cat in enumerate(self._valid_ids):
            cat_info.append({'name': cat, 'id': i + 1})
        ret = {'images': [], 'annotations': [], "categories": cat_info}

        self.max_objs = 2
        self.class_name = ['panel', 'cropped_panel']

        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                        for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))