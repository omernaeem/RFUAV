"""
基础模型
读模型要从nn.Module中的module类中继承
"""
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from collections import defaultdict
from logging import FileHandler
from typing import Iterable, List, Optional, Union

import torch
import torch.nn as nn
from .yolo import DetectionModel
from .yolo.basic import Ensemble, Detect
import io

class YOLOV5S(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self,
                 weights='yolov5s.pt',
                 device=torch.device('cpu'),
                 dnn=False,
                 data=None,
                 fp16=False,
                 fuse=True):

        super().__init__()
        model = Ensemble()
        w = str(weights[0] if isinstance(weights, list) else weights)
        fp16 &= False  # FP16
        nhwc = False  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        # load model
        model = DetectionModel()

        ckpt = model['model'].to(device).float()
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

        # Module compatibility updates
        for m in model.modules():
            t = type(m)
            if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect):
                m.inplace = True  # torch 1.7.0 compatibility
                if t is Detect and not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
            elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        model = model[-1]

        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)