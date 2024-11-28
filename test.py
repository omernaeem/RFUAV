import cv2
from PIL import Image
import numpy as np
import os


def read_image_with_chinese_path(image_path):

    pil_image = Image.open(image_path)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return cv_image


def crop_and_save_image(image,
                        output_path,
                        x,
                        y,
                        width,
                        height):
    """
    截取图像中固定位置的一个矩形窗，并将截取的内容保存到指定位置。

    Args:
        image (str): 输入图像的路径。
        output_path (str): 输出图像的路径。
        x (int): 矩形窗左上角的 x 坐标。
        y (int): 矩形窗左上角的 y 坐标。
        width (int): 矩形窗的宽度。
        height (int): 矩形窗的高度。

    Returns:
        None
    """
    # 截取矩形窗
    cropped_image = image[y:y+height, x:x+width]

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存截取的图像
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")


def CropImage(
        fig_save_path: str,
        file_path: str,
        x,
        y,
        width,
        height
):

    drones = os.listdir(file_path)

    for drone in drones:
        packlist = os.listdir(file_path + drone)
        for pack in packlist:

            _ = os.path.join(file_path+drone, pack)

            if os.path.isfile(_):
                check_folder(fig_save_path + drone)
                crop_and_save_image(read_image_with_chinese_path(_), fig_save_path + drone + '/' + pack, x=x, y=y, width=width, height=height)

            else:
                imgs = os.listdir(_)
                for img in imgs:
                    _save_path = os.path.join(fig_save_path + drone, pack)
                    check_folder(_save_path)
                    img_path = os.path.join(_, img)
                    crop_and_save_image(read_image_with_chinese_path(img_path), _save_path + '/' + img, x=x, y=y, width=width, height=height)
                    print(img + ' Done')
            print(pack + ' Done')
        print(drone + ' Done')
    print('All Done')


def check_folder(folder_path):
    """
    Checks and creates the folder if it does not exist.

    Parameters:
    - folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"folder '{folder_path}' created。")
    else:
        print(f"folder '{folder_path}' existed。")


def main():
    data_path = "E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/3.dataset-origin+20dB_defaultcolor_usingAG/"
    output_path = "E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/123/"
    x, y, width, height = 295, 140, 1710, 1460  # 矩形窗的坐标和尺寸
    # crop_and_save_image(read_image_with_chinese_path(image_path), output_path, x, y, width, height)

    CropImage(output_path, data_path, x, y, width, height)


if __name__ == '__main__':
    main()