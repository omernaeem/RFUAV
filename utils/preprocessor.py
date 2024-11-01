"""模型预处理方法:
1.给dataloader的预处理器 a.输入模型数据进行符合尺度的裁剪.
2.数据集用的数据增强方法.
3.原始数据的处理方法 a.加噪 b.复杂环境融合 c.滤波
"""

import cv2
import albumentations as A
import os
import logging


"""Augmentations for image data augmentation
All the augmentations methods are provided by albumentations: https://github.com/albumentations-team/albumentations

arg: 
    dataset_path(str): the file path of dataset
    methods(list[str]): the augmentation method, 
    default method using: 1.AdvancedBlur, 2.CLAHE, 3.ColorJitter, 4.GaussNoise, 5.ISONoise, 6.Sharpen
    you can find all the method in https://albumentations.ai/docs/api_reference/full_reference/.
    output_path(str)(optional): The augmented dataset will be saved here, if you are specialized.
     if the output is not specialized, function will create a new dir dataset_aug to store the new dataset in 
     the data_path
"""


def data_augmentation(dataset_path: str = None,
                      output_path: str = None,
                      methods: list[str] = None):

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset path does not exist")

    if not output_path:
        prefix = os.path.dirname(os.path.dirname(dataset_path))
        output_path = os.path.join(prefix, 'dataset_aug')
        os.mkdir(output_path)

    if methods is None:
        methods = [
            A.AdvancedBlur(
                blur_limit=(7, 13),
                sigma_x_limit=(7, 13),
                sigma_y_limit=(7, 13),
                rotate_limit=(-90, 90),
                beta_limit=(0.5, 8),
                noise_limit=(2, 10),
                p=1),
            A.CLAHE(
                clip_limit=3,
                tile_grid_size=(13, 13),
                p=1),
            A.ColorJitter(
                brightness=(0.5, 1.5),
                contrast=(1, 1),
                saturation=(1, 1),
                hue=(-0, 0),
                p=1),
            A.GaussNoise(
                var_limit=(100, 500),
                mean=0,
                p=1),
            A.ISONoise(
                intensity=(0.2, 0.5),
                color_shift=(0.01, 0.05),
                p=1),
            A.Sharpen(
                alpha=(0.2, 0.5),
                lightness=(0.5, 1),
                p=1)
            ]

    total_path = [
        os.path.join(dataset_path, 'train'),
        os.path.join(dataset_path, 'valid'),
    ]

    for path in total_path:
        _ = os.path.join(output_path, path.split('/')[-1])
        if not os.path.exists(_):
            os.mkdir(_)
        classes = os.listdir(path)

        for _class in classes:
            _save_path = os.path.join(_, _class)
            if not os.path.exists(_save_path):
                os.mkdir(_save_path)
            path_image = os.path.join(path, _class)
            images = os.listdir(path_image)
            i = 0

            for method in methods:

                for image in images:
                    original_image = cv2.cvtColor(cv2.imread(os.path.join(path_image, image)), cv2.COLOR_BGR2RGB)
                    transform = A.Compose(method)
                    augmented = transform(image=original_image)
                    cv2.imwrite(os.path.join(_save_path, image+'_AugM'+str(i)+os.path.splitext(image)[1]), augmented['image'])
                    cv2.imwrite(os.path.join(_save_path, image+'_origin'+os.path.splitext(image)[1]), original_image)
                i += 1

            logging.info('Finished augmentation of '+_class)



def show_image(image):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    data_path = "E:/Dataset_log/drone_thesis_classification_v2/"
    output_path = "E:/Drone_dataset/RFUAV/augmentation_exp2/"
    methods = [A.AdvancedBlur(
                    blur_limit=(7, 13),
                    sigma_x_limit=(7, 13),
                    sigma_y_limit=(7, 13),
                    rotate_limit=(-90, 90),
                    beta_limit=(0.5, 8),
                    noise_limit=(2, 10),
                    p=1),
               A.CLAHE(
                   clip_limit=3,
                   tile_grid_size=(13, 13),
                   p=1),
               A.ColorJitter(
                   brightness=(0.5, 1.5),
                   contrast=(1, 1),
                   saturation=(1, 1),
                   hue=(-0, 0),
                   p=1,
               ),
               A.GaussNoise(
                   var_limit=(100, 500),
                   mean=0,
                   p=1
               ),
               A.ISONoise(
                   intensity=(0.2, 0.5),
                   color_shift=(0.01, 0.05),
                   p=1
               ),
               A.Sharpen(
                   alpha=(0.2, 0.5),
                   lightness=(0.5, 1),
                   p=1
               )
               ]
    data_augmentation(dataset_path=data_path, output_path=output_path, methods=methods)


# test
if __name__ == '__main__':
    main()
