"""模型预处理方法:
1.给dataloader的预处理器 a.输入模型数据进行符合尺度的裁剪
2.数据集用的数据增强方法 对图像数据进行:a.
3.原始数据的处理方法 a.加噪 b.复杂环境融合 c.滤波
"""

import numpy as np
import cv2
import albumentations as A
import os


"""Augmentations for image data augmentation
All the augmentations methods are provided by albumentations: https://github.com/albumentations-team/albumentations

arg: 
    dataset_path(str): the file path of dataset
    methods(list[str]): the augmentation method.
    output_path(str): output path
"""


def data_augmentation(dataset_path, methods, output_path):
    output_path += 'dataset_aug'

    total_path = [
        os.path.join(dataset_path, 'train'),
        os.path.join(dataset_path, 'valid'),
    ]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for path in total_path:
        _ = os.path.join(output_path, path.split('/')[-1])
        if not os.path.exists(_):
            os.mkdir(_)

        images = os.listdir(path)
        i = 0
        for method in methods:
            for image in images:
                original_image = cv2.cvtColor(cv2.imread(os.path.join(path, image)), cv2.COLOR_BGR2RGB)
                transform = A.Compose(method)
                augmented = transform(image=original_image)
                cv2.imwrite(os.path.join(_, image+'_AugM'+str(i)+os.path.splitext(image)[1]), augmented['image'])
                cv2.imwrite(os.path.join(_, image+'_origin'+os.path.splitext(image)[1]), original_image)
            i += 1


def show_image(image):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    data_path = "E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/dataset/"
    output_path = "E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/res/"
    methods = [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2),]
    data_augmentation(data_path, methods, output_path)


# test
if __name__ == '__main__':
    main()
