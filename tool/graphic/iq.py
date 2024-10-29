"""
觀察二進制數據使用的工具
"""
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np
import os

fs = 100e6
stft_point = 2048
duration_time = 0.1
slice_point = int(fs * duration_time)

# 給批量畫圖用的路徑
fig_save_path = 'E:/Drone_dataset/RFUAV/pics_exp1_alldrones/'
file_path = 'E:/Drone_dataset/RFUAV/rawdata/'

# 給check用的路徑
datapack = 'E:/Drone_dataset/RFUAV/rawdata/DJFPVCOMBO/DJFPVCOMBO-16db-90db_5760m_100m_10m.iq'


def check_spectrum():
    drone_name = 'temp'
    with open(datapack, 'rb') as fp:

        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=np.float32)
        dataI = read_data[::2]
        dataQ = read_data[1::2]
        data = dataI + dataQ * 1j
        print('STFT transforming')

        f, t, Zxx = stft(data[0: slice_point],
                               fs, window=windows.hamming(stft_point), nperseg=stft_point, return_onesided=False)

        '''
        f_I, t_I, Zxx_I = stft(dataI[0: slice_point],
                               fs, window=windows.hamming(stft_point), nperseg=stft_point, noverlap=stft_point//2)


        f_Q, t_Q, Zxx_Q = stft(dataQ[0: slice_point],
                               fs, window=windows.hamming(stft_point), nperseg=stft_point, noverlap=stft_point // 2)
        
        
        print('Drawing')
        # I部分數據的時頻圖
        plt.figure()
        aug_I = 10 * np.log10(np.abs(Zxx_I))
        plt.pcolormesh(t_I, f_I, np.abs(aug_I), cmap='jet')
        plt.title(drone_name + " I")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        # plt.savefig(fig_save_path + drone_name + 'spectrum.png')
        plt.show()
        print("figure I done")

        # Q部分數據的時頻圖
        plt.figure()
        aug_Q = 10 * np.log10(np.abs(Zxx_Q))
        plt.pcolormesh(t_Q, f_Q, np.abs(aug_Q), cmap='jet')
        plt.title(drone_name + " Q")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        # plt.savefig(fig_save_path + drone_name + 'spectrum.png')
        plt.show()
        print("figure Q done")
        '''

        # 完整的時頻圖

        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)

        plt.figure()
        aug = 10 * np.log10(np.abs(Zxx))

        extent = [t.min(), t.max(), f.min(), f.max()]
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')

        # plt.pcolormesh(t, f, np.abs(aug), cmap='jet')

        plt.title(drone_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        # plt.savefig(fig_save_path + drone_name + 'spectrum.png')
        plt.show()
        print("figure done")



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
    check_spectrum()
    # DrawandSave()


if __name__ == '__main__':
    main()