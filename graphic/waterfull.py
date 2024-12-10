import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import fft
import cv2
import re
import h5py

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
    spectrogram = []

    pack_gap = 0
    j = 0
    gap = 150
    # 遍历每一帧数据
    for i in range(num_frames):
        # 获取当前帧的IQ数据，应用窗函数
        start_idx = i * (frame_size)
        frame_data = iq_data[start_idx:start_idx + frame_size] * window

        # 进行FFT转换到频域
        spectrum = fft(frame_data)
        spectrum = np.fft.fftshift(spectrum)  # 将频谱的零频率居中
        magnitude = np.abs(spectrum)  # 获取频谱的幅度
        if i > plot_size:
            spectrogram = spectrogram[1:]
            spectrogram = np.concatenate((spectrogram, magnitude.reshape(1, frame_size)), axis=0)

        else:
            spectrogram.append(magnitude)

        pack_gap += 1

        if i == plot_size:
            # 绘制瀑布图
            spectrogram = np.array(spectrogram)
            plt.figure()
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
            plt.axis('off')
            if output_dir != 'buffer':
                # 保存瀑布图
                plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.jpg'), dpi=300)
                plt.close()
            j += 1
            pack_gap =0

        if i > plot_size and pack_gap == gap:

            # 绘制瀑布图
            plt.figure()
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
            # 保存瀑布图
            plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.jpg'), dpi=300)
            plt.close()
            j += 1
            pack_gap = 0


def extract_number(s):
    # 使用正则表达式提取字符串中的数字部分
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return 0


def main():
    # 图片文件夹路径
    image_folder = 'C:/ML/RFUAV/res/'
    # 输出视频文件路径
    video_name = 'output_video.avi'

    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    # 按文件名排序
    images = sorted(images, key=extract_number)

    # 读取第一张图片以获取尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    # 写入每一帧
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # 释放视频写入对象
    cv2.destroyAllWindows()
    video.release()


# 示例调用
if __name__ == "__main__":

    # main()
    # 模拟IQ数据（假设是1024个点的信号，1个采样周期有2048个数据）
    fs = 100e6  # 采样率1MHz
    datapack = 'E:/Drone_dataset/RFUAV/rawdata/FutabaT14SG/FUtabaT14SG_2440_daifei_80dB(2)_0-2s.iq'
    save_path = 'E:/Drone_dataset/RFUAV/test/temp/'
    data = np.fromfile(datapack, dtype=np.float32)

    """
    data = h5py.File(datapack, 'r')
    data_I = data['RF0_I'][0]
    data_Q = data['RF0_Q'][0]
    data = data_I + data_Q * 1j
    """


    data = data[::2] + data[1::2] * 1j

    plot_waterfall_spectrogram(data, frame_size=256, fs=fs, output_dir=save_path, plot_size=39062)