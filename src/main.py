from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory


def main(opt):

  import torch
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter()
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  print('Model created...')
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  print('Trainer...')
  trainer = Trainer(opt, model, optimizer)
  print('Setting device...')
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up val data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    if opt.dataset == 'cityscapes':
        AP = val_loader.dataset.run_eval(preds, opt.save_dir)
        print('AP: ', AP)
    else:
        val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  print('Setting up train data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  # logger.write_model(model)
  print('Starting training...')
  best = 1e10
  best_AP = 0
  AP = 0
  import numpy as np
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    print('opt.input_h, opt.input_w:',opt.input_h, opt.input_w)
    mark = epoch if opt.save_all else 'last'
    
    log_dict_train, _ = trainer.train(epoch, train_loader, writer)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader, writer)
        if True and opt.dataset == 'cityscapes' and opt.task == 'polydet':
            AP = val_loader.dataset.run_eval(preds, opt.save_dir)
            print('AP: ', AP)
        
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if True and opt.dataset == 'cityscapes' and opt.task == 'polydet':
        logger.scalar_summary('AP', AP, epoch)
      if True and opt.dataset == 'cityscapes' and opt.task == 'polydet':
          if AP > best_AP:
              best_AP = AP
              save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                         epoch, model)
      else:
          if log_dict_val[opt.metric] < best:
            best = log_dict_val[opt.metric]
            save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                       epoch, model)
      writer.add_scalar("AP/val", AP, epoch)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                 epoch, model, optimizer)
      writer.add_scalar("AP/train", AP, epoch)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)