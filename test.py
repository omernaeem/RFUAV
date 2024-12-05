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




def plot_waterfall_spectrogram(iq_data, frame_size, fs, output_dir, plot_size):
    """
    绘制带颜色的瀑布图，并保存每一帧为图片
    :param iq_data: IQ信号数据 (复数形式)
    :param frame_size: 每一帧的大小
    :param fs: 采样频率
    :param output_dir: 保存图像的目录
    :param overlap_factor: 帧间重叠率（默认为50%）
    """
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frames = len(iq_data) // frame_size

    window = np.hanning(frame_size)


    j = 1
    # 遍历每一帧数据
    pt1 = 0
    pt2 = pt1 + plot_size
    while pt2 < len(iq_data):
        if pt1 == 0:
            # 获取当前帧的IQ数据，应用窗函数

            frame_data = iq_data[pt1: pt2] * window

            spectrum = fft(frame_data)
            spectrum = np.fft.fftshift(spectrum)
            magnitude = np.abs(spectrum)
            spectrogram = np.array(magnitude)

            # 绘制瀑布图
            plt.figure(figsize=(10, 6))
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])

            plt.axis('tight')
            plt.axis('off')
            # 保存瀑布图
            plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.png'))
            plt.close()

            pt1 += plot_size
            pt2 += plot_size


        else:
            # 获取当前帧的IQ数据，应用窗函数
            start_idx = i * (frame_size)
            frame_data = iq_data[start_idx:start_idx + frame_size] * window

            # 进行FFT转换到频域
            spectrum = fft(frame_data)
            spectrum = np.fft.fftshift(spectrum)  # 将频谱的零频率居中
            magnitude = np.abs(spectrum)  # 获取频谱的幅度

            # 将当前帧的频谱加入到瀑布图数据中
            spectrogram.append(magnitude)
            if i == plot_size * j:
                j += 1
                # 将所有帧的频谱转换为NumPy数组
                spectrogram = np.array(spectrogram)

                # 绘制瀑布图
                plt.figure(figsize=(10, 6))
                plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                           extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])

                plt.axis('tight')
                plt.axis('off')
                # 保存瀑布图
                plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.png'))
                plt.close()
                spectrogram = []

