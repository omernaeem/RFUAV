"""模型预处理方法:
1.给dataloader的预处理器 a.输入模型数据进行符合尺度的裁剪
2.数据集用的数据增强方法 对图像数据进行:a.
3.原始数据的处理方法 a.加噪 b.复杂环境融合 c.滤波
"""

import numpy as np
import cv2
import albumentations


"""Augmentations for image data augmentation
All the augmentations methods are provided by albumentations: https://github.com/albumentations-team/albumentations

"""

image = cv2.imread("")

