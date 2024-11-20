"""Augmentations for image data augmentation
All the augmentations methods are provided by albumentations: https://github.com/albumentations-team/albumentations

Args:
    dataset_path (str): The file path of the dataset.
    methods (list[str], optional): The augmentation methods to apply. Default methods include:
        1. AdvancedBlur
        2. CLAHE
        3. ColorJitter
        4. GaussNoise
        5. ISONoise
        6. Sharpen
    output_path (str, optional): The path where the augmented dataset will be saved. If not specified, a new directory named `dataset_aug` will be created in the same directory as the dataset.
"""

import cv2
import albumentations as A
import os


def data_augmentation(dataset_path: str = None,
                      output_path: str = None,
                      methods: list[str] = None):
    """
    Perform data augmentation on the given dataset using specified methods.

    Args:
        dataset_path (str): The file path of the dataset.
        output_path (str, optional): The path where the augmented dataset will be saved. If not specified, a new directory named `dataset_aug` will be created in the same directory as the dataset.
        methods (list, optional): The augmentation methods to apply. Each method is an instance of an Albumentations transformation. If not specified, default methods are used.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
    """

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
    data_augmentation(data_path, methods, output_path)


# test
if __name__ == '__main__':
    main()
