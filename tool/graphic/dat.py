import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import math


# 写一个截取算法，统一尺度，当采样率不够的时候要有一个截取算法，在频率上
# 要把所有文件用.dat来读，速度很快
1

# 不断读取文件夹内文件并拼接数据
# 检测前文件路径
sourceFolderPath = 'E:/360MoveData/Users/sam826001/Desktop/temp'
# 输出实文件夹路径
targetFolderPath = 'E:/360MoveData/Users/sam826001/Desktop/done'
figure_OutPath = 'E:/360MoveData/Users/sam826001/Desktop/1'
file_all = []
data_all = []
dataI = np.array([])
dataQ = np.array([])
frame = 1
time_duration = 0.01  # 0.03
fs = 100e6  # 15e6
slice_point = int(fs * time_duration)
stft_point = 2048
temp_dataI = np.array([])
temp_dataQ = np.array([])
temp_data = np.array([])
figure_doneI = 0
figure_doneQ = 0
file_flag = 0
read_flag = 1  # 读取的文件序号
# colors = [(0, 'blue'), (1, 'yellow')]  # (位置, 颜色)元组
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)



# 预设参数中大江无人机采样率为15MHZ持续时间为0.03
# 老数据集中的无人机采样率为100MHZ持续时间为0.1
while True:
    # 判断剩余数据是否足够画一幅图(I部分)
    if len(dataI) - figure_doneI * slice_point >= slice_point:
        # 绘制 dataI 的 stft 图像
        for ii in range(len(dataI) // slice_point):
            _, _, Zxx = stft(dataI[ii * slice_point:(ii + 1) * slice_point],
                             fs, window=windows.hamming(stft_point), nperseg=stft_point)
            plt.figure()
            plt.pcolormesh(Zxx)
            plt.title("I " + str(ii))
            plt.pause(0.01)
            plt.close()
            figure_doneI += 1

    # 获取预备文件夹内的文件列表
    re_files = os.listdir(sourceFolderPath)
    # 过滤出文件，而不包括文件夹
    re_files = [file for file in re_files if os.path.isfile(os.path.join(sourceFolderPath, file))]
    # 按修改时间排序文件列表
    re_files.sort(key=lambda x: os.path.getmtime(os.path.join(sourceFolderPath, x)), reverse=True)
    for file in re_files:
        filePath = os.path.join(sourceFolderPath, file)
        if os.path.getsize(filePath) != 0:
            # 移动文件
            shutil.move(filePath, targetFolderPath)

    # 遍历文件，检索所有文件名并更新
    files = os.listdir(targetFolderPath)
    # 过滤出文件，而不包括文件夹
    files = [file for file in files if os.path.isfile(os.path.join(targetFolderPath, file))]
    # 按修改时间排序文件列表
    files.sort(key=lambda x: os.path.getmtime(os.path.join(targetFolderPath, x)), reverse=True)

    # 在file_all中，老的在前（1）
    if not file_all:
        file_all.extend(files)
    elif len(files) != len(file_all):
        file_all.append(files[-1])

    # 拼接数据
    if read_flag <= len(file_all):
        # 计算每个文件大小以及读取次数，每次读取画一幅图的数据量
        filePath = os.path.join(targetFolderPath, file_all[read_flag - 1])
        fileSize = os.path.getsize(filePath)
        readtime = int(np.ceil(fileSize / 8 / slice_point))  # 每个数据点是4个字节，I和Q各一个
        with open(filePath, 'rb') as fp:
            for i in range(readtime):
                if i == readtime - 1:
                    read_data = np.fromfile(fp, dtype=np.float32, count=slice_point * 2)
                elif i == 0:
                    read_data = np.fromfile(fp, dtype=np.float32, count=slice_point * 2 - len(temp_data))
                    read_data = np.concatenate((temp_data, read_data))
                    temp_data = np.array([])
                else:
                    read_data = np.fromfile(fp, dtype=np.float32, count=slice_point * 2)
                dataI = read_data[::2]
                dataQ = read_data[1::2]

                if len(dataI) >= slice_point:
                    # 绘制 dataI 的 stft 图像
                    for ii in range(len(dataI) // slice_point):
                        f, t, Zxx = stft(dataI[ii * slice_point:(ii + 1) * slice_point],
                                         fs, window=windows.hamming(stft_point), nperseg=stft_point)
                        plt.figure()
                        # norm = colors.Normalize(vmin=np.min(np.abs(Zxx)), vmax=np.max(np.abs(Zxx)))
                        # plt.imshow(np.abs(Zxx), norm=norm)
                        # Zxx_symmetric = np.vstack([np.abs(Zxx), np.flip(np.abs(Zxx), axis=1)])
                        # reversed_f = -f[::-1]
                        # f = np.concatenate((reversed_f, f))
                        plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))  # 获取复数的绝对值
                        plt.colorbar()
                        plt.title("I " + str(ii))
                        plt.pause(0.01)
                        plt.close()
                        figure_doneI += 1
                else:
                    temp_data = read_data
        read_flag += 1

