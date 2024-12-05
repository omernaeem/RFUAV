import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import os


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

    j = 1
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
            np.concatenate((spectrogram, magnitude.reshape(1, frame_size)), axis=0)
        else:
            spectrogram.append(magnitude)

        if i == plot_size:
            # 绘制瀑布图
            spectrogram = np.array(spectrogram)
            plt.figure(figsize=(10, 6))
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])

            plt.axis('tight')
            plt.axis('off')
            # 保存瀑布图
            plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.png'))
            plt.close()
            j += 1

        if i > plot_size:

            # 绘制瀑布图
            plt.figure(figsize=(10, 6))
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])

            plt.axis('tight')
            plt.axis('off')
            # 保存瀑布图
            plt.savefig(os.path.join(output_dir, str(j) + 'waterfall_spectrogram.png'))
            plt.close()
            j += 1


def plot_waterfall_spectrogram1(iq_data, frame_size, fs, output_dir, plot_size):
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

    # 获取频率轴
    freq_axis = np.fft.fftfreq(frame_size, d=1 / fs)
    freq_axis = np.fft.fftshift(freq_axis)  # 频率轴居中，零频率位于中央

    # 计算帧间重叠样本数
    num_frames = (len(iq_data)) // (frame_size)

    # 汉宁窗口函数（可选）
    window = np.hanning(frame_size)

    # 存储所有帧的频谱数据
    spectrogram = []
    j = 1
    # 遍历每一帧数据
    for i in range(num_frames):
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
            # 将所有帧的频谱转换为NumPy数组-
            spectrogram = np.array(spectrogram)

            # 绘制瀑布图
            plt.figure(figsize=(10, 6))
            plt.imshow(np.log10(spectrogram.T), aspect='auto', cmap='jet', origin='lower',
                       extent=[0, num_frames * (frame_size) / fs, -fs / 2, fs / 2])
            plt.colorbar(label='Magnitude (dB)')
            plt.title('Waterfall Spectrogram')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')

            # 保存瀑布图
            plt.savefig(os.path.join(output_dir,str(j) + 'waterfall_spectrogram.png'))
            plt.close()

            spectrogram = []


# 示例调用
if __name__ == "__main__":
    # 模拟IQ数据（假设是1024个点的信号，1个采样周期有2048个数据）
    fs = 100e6  # 采样率1MHz
    datapack = 'E:/Drone_dataset/RFUAV/crop_data/DJFPVCOMBO/DJFPVCOMBO-16db-90db_5760m_100m_10m/DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat'
    save_path = 'E:/Drone_dataset/RFUAV/waterfull_test/'

    data = np.fromfile(datapack, dtype=np.float32)
    data = data[::2] + data[1::2] * 1j

    plot_waterfall_spectrogram(data, frame_size=256, fs=fs, output_dir=save_path, plot_size=39062)
