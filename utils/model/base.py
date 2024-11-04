"""Model_Base
模型的基础部件类定义
先把用到的基础库在这边封装一边，用得频繁的基础库(像conv，batch_norm这些)就删了
在这边封装完主要是为了加输出信息
"""
"""ToDo
给添加后的组件信息做一个组件信息的输出logger
"""
from torch import nn
from torchvision import models

class LayerNormalization(nn.LayerNorm):
   """
   nn.LayerNorm的库版本
   """
   def __init__(self, normalized_shape, eps=1e-12, elementwise_affine=True):
      super().__init__(normalized_shape, eps, elementwise_affine)

      
class SWAtten(models.swin_transformer.ShiftedWindowAttention):
   """
   nn.ShiftedWindowAttention的库版本
   """
   def __init__(self, dim, window_size, shift_size, num_heads, qkv_bias = True, proj_bias = True, attention_dropout = 0, dropout = 0):
      super().__init__(dim, window_size, shift_size, num_heads, qkv_bias, proj_bias, attention_dropout, dropout)

      
class Linear_layer(nn.Linear):
   """
   nn.Linear的库版本
   """
   def __init__(self, in_features, out_features, bias=True):
      super().__init__(in_features, out_features, bias)

      
