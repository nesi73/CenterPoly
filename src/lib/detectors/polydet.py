from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import polydet_decode
from models.utils import flip_tensor
from utils.post_process import polydet_post_process
from .base_detector import BaseDetector
import cv2
from utils.image import get_affine_transform, affine_transform

class PolydetDetector(BaseDetector):
  def __init__(self, opt):
    super(PolydetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      tick = time.time()
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()

      # fg = output['fg'].sigmoid_()
      # border_hm = output['border_hm'].sigmoid_()
      polys = output['poly']
      pseudo_depth = output['pseudo_depth']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      
      forward_time = time.time()
      dets = polydet_decode(hm, polys, pseudo_depth, reg=reg, cat_spec_poly=self.opt.cat_spec_poly, K=self.opt.K)

      print(time.time() - tick)
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1, fg=None):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = polydet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    # trans_input = get_affine_transform(meta['c'], meta['s'], 0, [meta['out_width'], meta['out_height']])
    # fg = cv2.warpAffine(fg, trans_input, (meta['out_width'], meta['out_height']), flags=cv2.INTER_LINEAR)

    length = 38  # len(dets[0][1][0]) #TODO: length 38?
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, length)
      dets[0][j][:, :4] /= scale
      # dets[0][j][:, 5:-1] /= scale
      dets[0][j][:, 5:-1] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
        soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]

    return results

  def debug(self, debugger, images, dets, output, scale=1):
    print('PolydetDetector')
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
