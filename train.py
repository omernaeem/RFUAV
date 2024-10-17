"""
先读cfg
结果保存路径
输入路径
batchsizes
epochs
model
断点训练



顺序:
modelloader和modelbuilder写好
要可以根据cfg变可选model


按照cfg的内容往dataloader里面加数据

dataloader
数据预处理
读标签和种类到内存，画一张数据集情况的图



configloader
"""

"""
把数据集和模型hyp设计在CFG中
模型的built以及dataloader的重构
还有一些hpy的重构
"""


import argparse  # 解析命令行的输入，下端的augment调用
import math  # python的math库
import os  # 文件处理库
import random  # 随机数模块
import subprocess  # 多进程处理
import sys  # 系统相关库
import time  # 时间的转换和访问
from copy import deepcopy  # 浅层 (shallow) 和深层 (deep) 复制操作
from datetime import datetime  # tensorboard的toolkit
from pathlib import Path  # 面向对象的文件系统路径


# 因为comet是一个可选的功能库，使用以下的方式可以将comet_ml库作为一个可选择的模块加入训练过程
# 即若comet_ml不存在，系统将会忽略comet_ml库
# comet库是一个用于深度学习数据管理、追踪和控制的模块
try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None


import numpy as np
import torch
import torch.distributed as dist  # torch中用来支持分布式训练的
import torch.nn as nn  # torch模块中对神经网络设定的核心模块
import yaml  # python用来读yaml文件的库，注意在python3.9之后的版本中yaml库不再可以支持任意类型的yaml文件读入
from torch.optim import lr_scheduler  # 学习率调整的库
from tqdm import tqdm  # 进度条库

"""
ToDo
"""
from utils.callbacks import Callbacks



# resolve()方法是获取当前正在执行的脚本文件的绝对路径,Path是pathlib库中的一个类，可以用来执行拆分，连接等等操作
# __file__特殊变量是当前文件的路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
# 将根目录添加到python解释器的搜索路径(sys.path)中
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# os.path.relpath()是pathlib中用来计算一个基准路径到另一个基准路径之间的相对路径的函数
# Path.cwd()是获取当前工作脚本的绝对路径的函数
# 这一步操作的结果是ROOT会被赋成YOLOv5的根路径到train函数工作路径之间的相对路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP，从val.py中导入用来计算loss值的函数
# 从..\models\experimental.py中导入用来attempt_load函数，整个函数是用来加载一些集成模型或是单个模型



"""
toDo
"""

from utils.dataloader import attempt_load
from utils.dataloader import create_dataloader
# 从..\models\yolo.py中导入用来定义分类器的class

from utils.model import Model  


# 导入数据预处理函数
from utils.preprocessor import preprocessor


# 导入命令行可视化模块，这里看看搞个实时图表
from utils.visual import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file,
                           check_img_size, check_suffix, check_yaml, colorstr, select_device,
                           increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_mutation, strip_optimizer,
                           yaml_save)


from utils.loss import ComputeLoss
from utils.metrics import fitness

from utils.plots import plot_evolve
from utils.torch_utils import (ModelEMA, de_parallel, smart_DDP, smart_optimizer, smart_resume, torch_distributed_zero_first)


def parse_opt(known=False):
   parser = argparse.ArgumentParser()
   parser.add_argument('--weights', type=str, default=ROOT / '', help='initial weights path')
   parser.add_argument('--cfg', type=str, default='', help='train config path')
   parser.add_argument('--epochs', type=int, default=200, help='training epochs')
   parser.add_argument('--batch-size', type=int, default=16, help='total batch size for GPUs')
   parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
   
   parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
   
   parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
   
   parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
   parser.add_argument('--name', default='exp', help='save to project/name')
   
   parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
   
   parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
   
   parser.add_argument('--seed', type=int, default=0, help='Global training seed')
   
   parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

   # Logger arguments
   parser.add_argument('--entity', default=None, help='Entity')
   parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
   parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

   return parser.parse_known_args()[0] if known else parser.parse_args()


def train(opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
   save_dir, epochs, batch_size, weights, data, cfg, resume = \
      Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.data, opt.cfg, opt.resume
   
   callbacks.run('on_pretrain_routine_start')

   # Directories
   w = save_dir / 'weights'  # weights dir
   w.parent.mkdir(parents=True, exist_ok=True)  # make dir
   last, best = w / '', w / ''  # 模型类别选择


   # Save run settings
   yaml_save(save_dir / 'opt.yaml', vars(opt))



   "是否要删除"
   # Loggers
   data_dict = None
   loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

   # Register actions
   for k in methods(loggers):
      callbacks.register_action(k, callback=getattr(loggers, k))

   # Process custom dataset artifact link
   data_dict = loggers.remote_dataset
   if resume:  # If resuming runs from remote artifact
      weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size



   # Config
   plots = True  # create plots
   cuda = device.type != 'cpu'

   init_seeds(opt.seed + 1, deterministic=True)
   
   data_dict = data_dict or check_dataset(data)  # check if None
   train_path, val_path = data_dict['train'], data_dict['validation']
   nc = int(data_dict['nc'])  # number of classes
   names = data_dict['names']  # class names


   """改成图像分类的数据集"""
   is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset


   # Model
   check_suffix(weights, '')  # check weights
   pretrained = weights.endswith('')
   if pretrained:

      """
      大改
      要把模型CFG和datasetCFG分开读，参考openmmlab
      """
      ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
      model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
      exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
      csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
      csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
      model.load_state_dict(csd, strict=False)  # load
      LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
   else:
      model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
   amp = check_amp(model)  # check AMP


   # Image size
   gs = max(int(model.stride.max()), 32)  # grid size (max stride)
   imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple


   # Optimizer
   nbs = 64  # nominal batch size
   accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing

   """往CFG里面重组一遍"""
   hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
   optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

   # Scheduler
   if opt.cos_lr:
      lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
   else:
      lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
   scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
   

   # Resume
   best_fitness, start_epoch = 0.0, 0
   if pretrained:
      if resume:
         best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, weights, epochs, resume)
      del ckpt, csd

   # SyncBatchNorm
   if opt.sync_bn and cuda:
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
      LOGGER.info('Using SyncBatchNorm()')

   # Trainloader
   train_loader, dataset = create_dataloader(train_path,
                                             imgsz,
                                             batch_size,
                                             gs,
                                             hyp=hyp,  # 重构
                                             augment=True,
                                             cache=None if opt.cache == 'val' else opt.cache,
                                             rect=opt.rect,
                                             image_weights=opt.image_weights,
                                             quad=opt.quad,
                                             prefix=colorstr('train: '),
                                             shuffle=True,
                                             seed=opt.seed)
   labels = np.concatenate(dataset.labels, 0)
   mlc = int(labels[:, 0].max())  # max label class
   assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

   # Process 0
   val_loader = create_dataloader(val_path,
                                 imgsz,
                                 batch_size,
                                 gs,
                                 hyp=hyp,  # 这个超参数要重构
                                 cache= opt.cache,
                                 rect=True,
                                 rank=-1,
                                 pad=0.5,
                                 prefix=colorstr('val: '))[0]

   if not resume:
      if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
      model.half().float()  # pre-reduce anchor precision

   callbacks.run('on_pretrain_routine_end', labels, names)

   """重构"""
   # Model attributes
   nl = de_parallel(model).model1[-1].nl  # number of detection layers (to scale hyps)
   hyp['box'] *= 3 / nl  # scale to layers
   hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
   hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
   hyp['label_smoothing'] = opt.label_smoothing
   model.nc = nc  # attach number of classes to model
   model.hyp = hyp  # attach hyperparameters to model
   model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
   model.names = names

   # Start training
   t0 = time.time()
   nb = len(train_loader)  # number of batches

   last_opt_step = -1
   
   """评估参数换一下"""
   maps = np.zeros(nc)  # mAP per class
   results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
   
   
   scheduler.last_epoch = start_epoch - 1  # do not move
   scaler = torch.cuda.amp.GradScaler(enabled=amp)
   
   
   compute_loss = ComputeLoss(model)  # init loss class
   callbacks.run('on_train_start')
   LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
               f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
               f"Logging results to {colorstr('bold', save_dir)}\n"
               f'Starting training for {epochs} epochs...')
   
   
   for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
      callbacks.run('on_train_epoch_start')
      model.train()

      # Update image weights (optional, single-GPU only)
      if opt.image_weights:
         cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
         iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
         dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

      # Update mosaic border (optional)
      # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
      # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

      mloss = torch.zeros(3, device=device)  # mean losses

      pbar = enumerate(train_loader)
      LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
      pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
      optimizer.zero_grad()
      
      for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
         callbacks.run('on_train_batch_start')
         ni = i + nb * epoch  # number integrated batches (since train start)
         imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

         """这里的方法要重构到数据集增强方法中"""
         # Multi-scale  
         if opt.multi_scale:
               sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
               sf = sz / max(imgs.shape[2:])  # scale factor
               if sf != 1:
                  ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                  imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

         # Forward
         with torch.cuda.amp.autocast(amp):
               pred = model(imgs)  # forward
               loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
               if opt.quad:
                  loss *= 4.

         # Backward
         scaler.scale(loss).backward()

         # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
         if ni - last_opt_step >= accumulate:
               scaler.unscale_(optimizer)  # unscale gradients
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
               scaler.step(optimizer)  # optimizer.step
               scaler.update()
               optimizer.zero_grad()
               last_opt_step = ni

         # Log
         mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
         mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
         pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                              (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
         callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
         if callbacks.stop_training:
            return
         # end batch ------------------------------------------------------------------------------------------------

      # Scheduler
      lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
      scheduler.step()
      
      
      """大改"""
      if RANK in {-1, 0}:
         # mAP
         callbacks.run('on_train_epoch_end', epoch=epoch)
         ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
         final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
         if not noval or final_epoch:  # Calculate mAP
               results, maps, _ = validate.run(data_dict,
                                             batch_size=batch_size // WORLD_SIZE * 2,
                                             imgsz=imgsz,
                                             half=amp,
                                             model=ema.ema,
                                             single_cls=single_cls,
                                             dataloader=val_loader,
                                             save_dir=save_dir,
                                             plots=False,
                                             callbacks=callbacks,
                                             compute_loss=compute_loss)

         # Update best mAP
         fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
         stop = stopper(epoch=epoch, fitness=fi)  # early stop check
         if fi > best_fitness:
               best_fitness = fi
         log_vals = list(mloss) + list(results) + lr
         callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
         """上面的都要重构"""
         # Save model

         """又跟模型类别选择有关系"""
         if  final_epoch:  # if save
               ckpt = {
                  'epoch': epoch,
                  'best_fitness': best_fitness,
                  'model': deepcopy(de_parallel(model)).half(),
                  'ema': deepcopy(ema.ema).half(),
                  'updates': ema.updates,
                  'optimizer': optimizer.state_dict(),
                  'opt': vars(opt),
                  'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                  'date': datetime.now().isoformat()}

               # Save last, best and delete
               torch.save(ckpt, last)
               if best_fitness == fi:
                  torch.save(ckpt, best)
               if opt.save_period > 0 and epoch % opt.save_period == 0:
                  torch.save(ckpt, w / f'epoch{epoch}.pt')
               del ckpt
               callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)


      # end epoch ----------------------------------------------------------------------------------------------------
   # end training -----------------------------------------------------------------------------------------------------
   LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
   for f in last, best:
      if f.exists():
            strip_optimizer(f)  # strip optimizers
            if f is best:
               LOGGER.info(f'\nValidating {f}...')
               results, _, _ = validate.run(
                  data_dict,
                  batch_size=batch_size // WORLD_SIZE * 2,
                  imgsz=imgsz,
                  model=attempt_load(f, device).half(),
                  iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                  single_cls=single_cls,
                  dataloader=val_loader,
                  save_dir=save_dir,
                  save_json=is_coco,
                  verbose=True,
                  plots=plots,
                  callbacks=callbacks,
                  compute_loss=compute_loss)  # val best model with plots
               if is_coco:
                  callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

      callbacks.run('on_train_end', last, best, epoch, results)

   torch.cuda.empty_cache()
   return results


def main(opt, callbacks=Callbacks()):
   
   # Resume (from specified or most recent last.pt)
   if opt.resume:
      last = Path(check_file(opt.resume))

      # check
      opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml

      
      opt_data = opt.data  # original dataset
      
      with open(opt_yaml, errors='ignore') as f:
            d = yaml.safe_load(f)
      
      opt = argparse.Namespace(**d)  # replace
      opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate

   else:
      opt.data, opt.cfg, opt.weights, opt.project = \
         check_file(opt.data), check_yaml(opt.cfg), str(opt.weights), str(opt.project)  # checks
      assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
      
      opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

   device = select_device(opt.device, batch_size=opt.batch_size)

   # Train
   train(opt, device, callbacks)


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)