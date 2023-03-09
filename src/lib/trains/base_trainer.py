from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # if batch['input'].shape[1] != 3:
        #   batch['input'] = batch['input'][:, 6:9, :, :]
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

        # for name, param in model.named_parameters():
        #   if param.requires_grad:
        #       print(name)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)
    
    def add_coco_bbox(self, bbox, cat, img, conf=1, show_txt=True, img_id='default'): 
      bbox = np.array(bbox, dtype=np.int)
      # cat = (int(cat) + 1) % 80
      cat = int(cat)
      # print('cat', cat, self.names[cat])
      colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
      colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
      colors = colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      colors = np.clip(colors, 0., 0.6 * 255).astype(np.uint8)
      c = colors[cat][0][0].tolist()
      c = (255 - np.array(c)).tolist()

      txt ='panel' if cat == 1 else 'cropped_panel'
      font = cv2.FONT_HERSHEY_SIMPLEX
      cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
      print('bbox:', bbox)
      print('bbox:', bbox[0])
      print('type bbox:', type(bbox))
      print('type img:', type(img))
      print('shape img:', img.shape)
      c=(0,0,255)
      print('color:', c)
      m = np.ascontiguousarray(img, dtype=np.uint8)
      print('shape m:', m.shape)
      if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
          return
      cv2.rectangle(m, (bbox[0], bbox[1]), (bbox[2],bbox[3]), c, 2)
      if show_txt:
        cv2.rectangle(m,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
        cv2.putText(m, txt, (bbox[0], bbox[1] - 2), 
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
      
      cv2.imwrite('result_gt.jpg', img)
    
    def gen_colormap(self, img, output_res=None):
      colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
      colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
      colors = colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      colors = np.clip(colors, 0., 0.6 * 255).astype(np.uint8)
      img = img.copy()
      c, h, w = img.shape[0], img.shape[1], img.shape[2]
      if output_res is None:
        output_res = (h * 4, w * 4)
      img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
      colors = np.array(
        colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
      colors = 255 - colors
      color_map = (img * colors).max(axis=2).astype(np.uint8)
      color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
      return color_map

    def add_blend_img(self, back, fore, ax, img_id='blend', trans=0.7):
      fore = 255 - fore
      if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
        fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
      if len(fore.shape) == 2:
        fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
      final_img = (back * (1. - trans) + fore * trans)
      final_img[final_img > 255] = 255
      final_img[final_img < 0] = 0
      final_img = final_img.astype(np.uint8).copy()
      ax.imshow(final_img)
      ax.set_title('{}.jpg'.format(img_id))
      cv2.imwrite('{}.jpg'.format(img_id), final_img)

    def run_epoch(self, phase, epoch, data_loader, writer):
        DEBUG=False
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()

        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            import copy
            original_batch=copy.copy(batch)
            #convert Tensor to image
            if DEBUG:
              img = batch['input'][0].detach().cpu().numpy().transpose(1, 2, 0)
              img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
              print('shape img', img.shape)
              cv2.imwrite('img.jpg', img)
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(
                        device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + \
                    '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                a=batch['meta']['gt_det'].numpy()
                # self.debug(batch, output, iter_id)

            if opt.test or (True and opt.dataset == 'cityscapes' and phase == 'val'):
                self.save_result(output, batch, results)
            # dets_gt = batch['meta']['gt_det'].numpy()
            # print('len(dets_gt[0])',len(dets_gt[0]))
            # for k in range(len(dets_gt[0])):
            #   self.add_coco_bbox(dets_gt[0, k, :4], dets_gt[0, k, -1], img, img_id='out_gt')

            if DEBUG:        
              pred = self.gen_colormap(output['hm'][0].detach().cpu().numpy())
              gt = self.gen_colormap(batch['hm'][0].detach().cpu().numpy())
              f, axarr = plt.subplots(3,1) 
              axarr[0].imshow(img)
              axarr[0].set_title('{}.jpg'.format('original'))
              self.add_blend_img(img, pred, axarr[1], 'pred_hm')
              self.add_blend_img(img, gt,axarr[2], 'gt_hm')
              plt.draw()
              plt.waitforbuttonpress(0)
              plt.close()    
            
            writer.add_scalar("Loss/{}".format(phase), loss, epoch)
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, writer):
        return self.run_epoch('val', epoch, data_loader, writer)

    def train(self, epoch, data_loader, writer):
        return self.run_epoch('train', epoch, data_loader, writer)

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255