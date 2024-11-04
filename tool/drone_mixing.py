"""把两个或多个无人机的原始数据混合在一起，组成复杂无人机通讯环境
ToDo
封装成函数
"""

from scipy.io import loadmat
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import stft, windows


UAV1 = 'E:/Drone_dataset/RFA/DroneRFa/RadioLinkAT9S_FLY/high/done_jet/T10101_S1000.mat'
UAV2 = 'E:/Drone_dataset/RFA/DroneRFa/FutabaT14SG_FLY/high/done_jet/T10110_S1000.mat'


time_duration = 0.1  # 预设时间窗的长度(关键参数)，决定x尺度
fs = 100e6  # 采样带宽要和采样设备匹配,实验室设备的采样带宽一般为15MHZ，原论文中的最大采样带宽为100MHZ，采样带宽决定y轴尺度
slice_point = int(fs*time_duration)

fly_name = 'Multi Fly'  # 预设无人机的名称
sourceFolderPath = 'E:/360MoveData/Users/sam826001/Desktop/AVATA/'  # 预设读取的目录
targetFolderPath = 'C:/Users/user/Desktop/clip2/'  # 预设保存的目录

stft_point = 2048  # 采样点数是每次y轴用的点数,1024就够用了


def main():
    # 设置一个循环使程序可以连续的读取一个文件中的所有图片
    data_UAV1 = h5py.File(UAV1, 'r')
    data_UAV2 = h5py.File(UAV2, 'r')

    # data_UAV1_I1 = data_UAV1['RF0_I'][0]
    data_UAV1_I2 = data_UAV1['RF0_I'][0]


    data_UAV2_I1 = data_UAV2['RF0_I'][0]  # 同一组信号的IQ读一路就行
    # data_UAV2_I2 = data_UAV2['RF1_I'][0]  # 同一组信号的IQ读一路就行

    # 获取两个数组的长度
    len1 = len(data_UAV1_I2)
    len2 = len(data_UAV2_I1)

    # 确定最长的长度
    max_len = max(len1, len2)

    # 将较短的数组用0补齐
    data_UAV1_I2_padded = np.pad(data_UAV1_I2, (0, max_len - len1), 'constant', constant_values=0)
    data_UAV2_I1_padded = np.pad(data_UAV2_I1, (0, max_len - len2), 'constant', constant_values=0)

    # 对齐后的数组进行逐元素相加
    data_merged = data_UAV1_I2_padded + data_UAV2_I1_padded

    j = 0
    i = 0
    while (j+1)*slice_point <= len(data_merged):

        """
        f_I2, t_I2, Zxx_I2 = stft(data_UAV1_I2[i*slice_point: (i+1) * slice_point],
                         fs, window=windows.hamming(stft_point), nperseg=stft_point)
        augmentation_Zxx2 = 20*np.log10(np.abs(Zxx_I2))  # 是否选择增强
        plt.figure()
        plt.pcolormesh(t_I2, f_I2, augmentation_Zxx2, cmap='jet')
        plt.title(fly_name + (str(i)))
        plt.savefig(targetFolderPath + fly_name + '_fir_' + str(i) + '_20-40m_' + '2.4GHZ.jpg', dpi=300)
        plt.close()

        f2_I1, t_I1, Zxx_I1 = stft(data_UAV2_I1[i*slice_point: (i+1) * slice_point],
                         fs, window=windows.hamming(stft_point), nperseg=stft_point)
        augmentation_2_Zxx1 = 20*np.log10(np.abs(Zxx_I1))
        plt.figure()
        plt.ioff()
        plt.pcolormesh(t_I1, f2_I1, augmentation_2_Zxx1, cmap='jet')
        plt.title(fly_name + (str(i)))
        plt.savefig(targetFolderPath + fly_name + '_sec_' + str(i) + '_20-40m_' + '985MHZ.jpg', dpi=300)  # dpi设置了图片保存的清晰度，300对应1K的图片
        plt.close()
        """


        fm_I1, tm_I1, Zxxm_I1 = stft(data_merged[i*slice_point: (i+1) * slice_point],
                         fs, window=windows.hamming(stft_point), nperseg=stft_point)
        augmentation_merge_Zxx1 = 20*np.log10(np.abs(Zxxm_I1))
        # 画第一组IQ
        plt.figure()
        plt.ioff()  # 关闭可视窗
        plt.pcolormesh(tm_I1, fm_I1, augmentation_merge_Zxx1, cmap='jet')
        plt.axis('off')
        plt.title('Multi Drone')
        plt.savefig(targetFolderPath + fly_name + '' + str(i), dpi=300)
        plt.close()

        i += 1
        j += 1

    return 0


if __name__ == '__main__':
    main()