import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np
import os

fs = 100e6   # 实验室里的USRP最适合24MHZ加0.1s，雷达采的采样率为30MHZ
stft_point = 1024
duration_time = 0.1
slice_point = int(fs * duration_time)

fig_save_path = 'E:/Drone_dataset/RFUAV/pics_exp1_alldrones/'
file_path = 'E:/Drone_dataset/RFUAV/rawdata/'
datapack = 'E:/Drone_dataset/RFUAV/UAVDATA/DJI AVATA2/DJI AVTA2-SNR0dB-85db_5765m_100m_10m(3).iq'


def check_spectrum():
    drone_name = ''
    with open(datapack, 'rb') as fp:
        read_data = np.fromfile(fp, dtype=np.int16)
        data = read_data[::2]
        f, t, Zxx = stft(data[0: slice_point],
                         fs, window=windows.hamming(stft_point), nperseg=stft_point, noverlap=stft_point//2)
        plt.figure()
        aug = 20 * np.log10(np.abs(Zxx))
        plt.pcolormesh(t, f, np.abs(aug), cmap='jet')
        plt.title(drone_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(fig_save_path + drone_name + 'spectrum.png')
        plt.show()


def DrawandSave():
        re_files = os.listdir(file_path)
        for file in re_files:
            packlist = os.listdir(file_path + file)
            for pack in packlist:

                check_folder(fig_save_path + file + '/' + pack)

                packname = os.path.join(file_path + file, pack)
                read_data = np.fromfile(packname, dtype=np.float32)
                data = read_data[::2]
                i = 0
                j = 0
                while (j+1)*slice_point <= len(data):
                    f_I, t_I, Zxx_I = stft(data[i*slice_point: (i+1) * slice_point],
                                     fs, window=windows.hamming(stft_point), nperseg=stft_point)
                    augmentation_Zxx1 = 20*np.log10(np.abs(Zxx_I))
                    plt.ioff()
                    plt.pcolormesh(t_I, f_I, augmentation_Zxx1)
                    plt.title(file)
                    plt.savefig(fig_save_path + file + '/' + pack + '/' + file + ' (' + str(i) + ').jpg', dpi=300)
                    plt.close()
                    i += 1
                    j += 1
                    print(pack + ' Done')
                print(file + ' Done')
            print('All Done')


def check_folder(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夾 '{folder_path}' 已創建。")
    else:
        print(f"文件夾 '{folder_path}' 已存在。")


def main():
    # check_spectrum()
    DrawandSave()


if __name__ == '__main__':
    main()