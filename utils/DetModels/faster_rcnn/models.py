import torch
from torch import nn


class Faster_RCNN(nn.Module):
    def __init__(self,
                 weights='faster_rcnn_resnet50_fpn_coco.pth',
                 device=torch.device('cpu'),
                 dnn=False,
                 data=None,
                 fp16=False,
                 fuse=True):
        super().__init__()
