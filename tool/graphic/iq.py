"""观察时频图的工具
包含: 1.一个观察一包数据前0.1s时频图的便捷观察工具
     2. 把一包数据画成视频的工具
     3.批量画图工具

________________
1.把默认的测试路径补上

"""
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np
import os
from io import BytesIO
import imageio
from PIL import Image
import queue
import threading
import time


def generate_images(datapack: str,
                    file: str,
                    pack: str,
                    q: queue.Queue,
                    fs: int,
                    stft_point: int,
                    duration_time: float,
                    ratio: int = 1,  # 控制产生图片时间间隔的倍率，默认为3生成视频的倍率
                    location: str = 'buffer',
                    ):

    time1 = time.time()
    slice_point = int(fs * duration_time)
    read_data = np.fromfile(datapack, dtype=np.float32)
    data = read_data[::2] + read_data[1::2] * 1j

    i = 0
    while (i + 1) * slice_point <= len(data):
        f, t, Zxx = STFT(data[int(i * slice_point): int((i + 1) * slice_point)], stft_point=stft_point, fs=fs, duration_time=duration_time)
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]

        plt.figure()
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

        if location == 'buffer':
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300)
            plt.close()

            buffer.seek(0)
            image = Image.open(buffer)
            image_array = np.array(image)

            q.put(image_array)

        else:
            plt.savefig(location + file + '/' + pack + '/' + file + ' (' + str(i) + ').jpg', dpi=300)

        i += 2 ** (-ratio)


def manage_cache(q: queue.Queue, save_path: str, cache_size: int, fps: int):
    all_images = []
    while True:
        try:
            image = q.get(timeout=1)  # 非阻塞地从队列中获取图像
            all_images.append(image)

            if len(all_images) >= cache_size or q.empty():
                video_path = save_path + 'video.mp4'
                imageio.mimsave(video_path, all_images, fps=fps)
                all_images = []
                print(f"Saved video to {video_path}")
        except queue.Empty:
            if q.empty():
                break  # 如果队列为空且没有新的图像生成，退出循环


def save_as_video(datapack: str,
                  save_path: str,
                  fs: int = 100e6,
                  stft_point: int = 1024,
                  duration_time: float = 0.1,
                  cache_size: int = 50,  # 缓存大小，单位为图像数量
                  fps: int = 15  # 视频帧率
                  ):
    start = time.time()
    q = queue.Queue(maxsize=cache_size)

    # 创建生成图像的线程
    generator_thread = threading.Thread(target=generate_images, args=(datapack, None, None,  q, fs, stft_point, duration_time))
    generator_thread.start()

    # 创建管理缓存的线程
    manager_thread = threading.Thread(target=manage_cache, args=(q, save_path, cache_size, fps))
    manager_thread.start()

    # 等待两个线程完成
    generator_thread.join()
    manager_thread.join()

    print(f"Total time: {time.time() - start} seconds")


def show_spectrum(datapack: str = '',
                  drone_name: str = 'test',
                  fs: int = 100e6,
                  stft_point: int = 2048,
                  duration_time: float = 0.1,
                  Middle_Frequency: float = 2400e6,
                  ):

    with open(datapack, 'rb') as fp:
        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=np.float32)

        data = read_data[::2] + read_data[1::2] * 1j
        print('STFT transforming')

        f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs, duration_time=duration_time)
        f = np.linspace(Middle_Frequency-fs / 2, Middle_Frequency+fs / 2, stft_point)
        Zxx = np.fft.fftshift(Zxx, axes=0)

        plt.figure()
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(drone_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        plt.show()
        print("figure done")


def show_half_only(datapack: str = '',
                   drone_name: str = 'test',
                   fs: int = 100e6,
                   stft_point: int = 2048,
                   duration_time: float = 0.1,
                   ):
    with open(datapack, 'rb') as fp:
        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=np.float32)
        dataI = read_data[::2]
        dataQ = read_data[1::2]

        f_I, t_I, Zxx_I = STFT(dataI, fs=fs, stft_point=stft_point, duration_time=duration_time)
        f_Q, t_Q, Zxx_Q = STFT(dataQ, fs=fs, stft_point=stft_point, duration_time=duration_time)

        # I部分數據的時頻圖
        print('Drawing')
        plt.figure()
        aug_I = 10 * np.log10(np.abs(Zxx_I))
        plt.pcolormesh(t_I, f_I, np.abs(aug_I))
        plt.title(drone_name + " I")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.show()
        print("figure I done")

        # Q部分數據的時頻圖
        plt.figure()
        aug_Q = 10 * np.log10(np.abs(Zxx_Q))
        plt.pcolormesh(t_Q, f_Q, np.abs(aug_Q), cmap='jet')
        plt.title(drone_name + " Q")
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.show()
        print("figure Q done")


def DrawandSave(
        fig_save_path: str,
        file_path: str,
        fs: int = 100e6,
        stft_point: int = 2048,
        duration_time: float = 0.1,
):
    slice_point = int(fs * duration_time)
    re_files = os.listdir(file_path)

    for file in re_files:
        packlist = os.listdir(file_path + file)
        for pack in packlist:

            check_folder(fig_save_path + file + '/' + pack)

            packname = os.path.join(file_path + file, pack)
            read_data = np.fromfile(packname, dtype=np.float32)
            data = read_data[::2]
            j = 0
            while (j + 1) * slice_point <= len(data):
                generate_images(datapack=data[int(j * slice_point): int((j + 1) * slice_point)],
                                file=file,
                                pack=pack,
                                ratio=0,
                                stft_point=stft_point,
                                location=fig_save_path,
                                )

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


def STFT(data,
         stft_point: int = 1024,
         fs: int = 100e6,
         duration_time: float = 0.1
         ):

    slice_point = int(fs * duration_time)

    f, t, Zxx = stft(data[0: slice_point],
         fs, window=windows.hamming(stft_point), nperseg=stft_point)

    return f, t, Zxx


# test----------------------------------------------------------------------------------------
def main():
    datapack = 'E:/Drone_dataset/RFUAV/crop_data/DJFPVCOMBO/DJFPVCOMBO-16db-90db_5760m_100m_10m/DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat'
    save_path = 'E:/Drone_dataset/RFUAV/darw_test/'
    save_as_video(datapack=datapack,
                  save_path=save_path,
                  fs=100e6,
                  stft_point=1024,
                  duration_time=0.1,
                  cache_size=50,
                  fps=0.1,
                  )
    """
    show_spectrum(datapack=datapack,
                  drone_name='test',
                  fs=100e6,
                  stft_point=2048,
                  duration_time=0.1,
                  )
    show_half_only(datapack, 
                   drone_name='test',
                   fs=100e6,
                   stft_point=2048,
                   duration_time=0.1,
                   )
    # DrawandSave()
    """


if __name__ == '__main__':
    main()