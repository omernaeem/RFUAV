# Tool to process raw data
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import numpy as np
import os
from io import BytesIO
import imageio
from PIL import Image
from typing import Union


class RawDataProcessor:
    """transform raw data into images, video, and save the result locally
    func:
    - TransRawDataintoSpectrogram() can process raw data in batches and save the results locally
    - TransRawDataintoVideo() can transform raw data into video and save the result locally
    - ShowSpectrogram() can show the spectrum of the raw data in prev 0.1s to check the raw data quicly
    """

    def TransRawDataintoSpectrogram(self,
                                    fig_save_path: str,
                                    data_path: str,
                                    sample_rate: Union[int, float] = 100e6,
                                    stft_point: int = 2048,
                                    duration_time: float = 0.1,
                                    ):
        """transform the raw data into spectromgrams and save the results locally
        :param fig_save_path: the target dir path to save the image result.
        :param data_path: the input raw data dir path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer).
        :param duration_time: the duration time of single spectromgram.
        """
        DrawandSave(fig_save_path=fig_save_path, file_path=data_path, fs=sample_rate,
                    stft_point=stft_point, duration_time=duration_time)

    def TransRawDataintoVideo(self,
                              save_path: str,
                              data_path: str,
                              sample_rate: Union[int, float] = 100e6,
                              stft_point: int = 2048,
                              duration_time: float = 0.1,
                              fps: int = 5
                              ):
        """transform the raw data into video and save the result locally
        :param save_path: the target dir path to save the image result.
        :param data_path: the input raw data dir path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer). ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
        :param duration_time: the duration time of single spectromgram.
        :param fps: control the fps of generated video.
        """
        save_as_video(datapack=data_path, save_path=save_path, fs=sample_rate,
                      stft_point=stft_point, duration_time=duration_time, fps=fps)

    def ShowSpectrogram(self,
                        data_path: str,
                        drone_name: str = 'test',
                        sample_rate: Union[int, float] = 100e6,
                        stft_point: int = 2048,
                        duration_time: float = 0.1,
                        oneside: bool = False,
                        Middle_Frequency: float = 2400e6
                        ):
        """tool used to observe the spectrograms from a local datapack
        :param save_path: the target dir path to save the image result.
        :param data_path: the input raw data file path.
        :param sample_rate: the simple rate when collect the raw frequency data using USRP, ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
        :param stft_point: the STFT points using in STFT transformation, you better set this to 2**n (n is a Non-negative integer). ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
        :param duration_time: the duration time of single spectromgram.
        :param oneside: set 'True' if you want to observe the real & imaginary parts separately, set 'False' to show the complete spectrogram
        :param Middle_Frequency: the middle frequency set to collect data using USRP in the frequency band, ref: https://en.wikipedia.org/wiki/Center_frequency
        """

        if oneside:
            show_half_only(datapack=data_path, drone_name=drone_name,
                           fs=sample_rate, stft_point=stft_point, duration_time=duration_time)

        else:
            show_spectrum(datapack=data_path, drone_name=drone_name, fs=sample_rate, stft_point=stft_point,
                           duration_time=duration_time, Middle_Frequency=Middle_Frequency)


def generate_images(datapack: str = None,
                    file: str = None,
                    pack: str = None,
                    fs: int = 100e6,
                    stft_point: int = 1024,
                    duration_time: float = 0.1,
                    ratio: int = 1,  # 控制产生图片时间间隔的倍率，默认为1生成视频的倍率
                    location: str = 'buffer',
                    ):
    """
    Generates images from the given data using Short-Time Fourier Transform (STFT).

    Parameters:
    - datapack (str): Path to the data file.
    - file (str): File name.
    - pack (str): Pack name.
    - fs (int): Sampling frequency, default is 100 MHz. ref: https://en.wikipedia.org/wiki/Sampling_(signal_processing)
    - stft_point (int): Number of points for STFT, default is 1024. ref: https://en.wikipedia.org/wiki/Short-time_Fourier_transform#Inverse_STFT
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - ratio (int): Controls the time interval ratio for generating images, default is 1.
    - location (str): Location to save the images, default is 'buffer'.

    Returns:
    - list: List of images if `location` is 'buffer'.
    """
    slice_point = int(fs * duration_time)
    read_data = np.fromfile(datapack, dtype=np.float32)
    data = read_data[::2] + read_data[1::2] * 1j
    images = []

    i = 0
    while (i + 1) * slice_point <= len(data):

        f, t, Zxx = STFT(data[int(i * slice_point): int((i + 1) * slice_point)],
                         stft_point=stft_point, fs=fs, duration_time=duration_time, onside=False)
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
            images.append(Image.open(buffer))

        else:
            plt.savefig(location + (file + '/') if file else '' + (pack + '/') if pack else '' + ' (' + str(i) + ').jpg', dpi=300)
            plt.close()

        i += 2 ** (-ratio)

    if location == 'buffer':
        return images


def save_as_video(datapack: str,
                  save_path: str,
                  fs: int = 100e6,
                  stft_point: int = 1024,
                  duration_time: float = 0.1,
                  fps: int = 5  # 视频帧率
                  ):
    """
    Saves the generated images as a video.

    Parameters:
    - datapack (str): Path to the data file.
    - save_path (str): Path to save the video.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 1024.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - fps (int): Frame rate of the video, default is 5.
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(datapack):
        raise ValueError('File not found!')

    images = generate_images(datapack=datapack, fs=fs, stft_point=stft_point, duration_time=duration_time)
    imageio.mimsave(save_path+'video.mp4', images, fps=fps)


def show_spectrum(datapack: str = '',
                  drone_name: str = 'test',
                  fs: int = 100e6,
                  stft_point: int = 2048,
                  duration_time: float = 0.1,
                  Middle_Frequency: float = 2400e6,
                  ):

    """
    Displays the spectrum of the given data.

    Parameters:
    - datapack (str): Path to the data file.
    - drone_name (str): Name of the drone, default is 'test'.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    - Middle_Frequency (float): Middle frequency, default is 2400 MHz.
    """

    with open(datapack, 'rb') as fp:
        print("reading raw data...")
        read_data = np.fromfile(fp, dtype=np.float32)

        data = read_data[::2] + read_data[1::2] * 1j
        print('STFT transforming')

        f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs, duration_time=duration_time, onside=False)
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


def show_half_only(datapack: str = '',
                   drone_name: str = 'test',
                   fs: int = 100e6,
                   stft_point: int = 2048,
                   duration_time: float = 0.1,
                   ):

    """
    Displays I and Q components of the given data separately.

    Parameters:
    - datapack (str): Path to the data file.
    - drone_name (str): Name of the drone, default is 'test'.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.
    """

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

    """
    Draw and save the images from the given data files.

    Parameters:
    - fig_save_path (str): Path to save the figures.
    - file_path (str): Path to the data files.
    - fs (int): Sampling frequency, default is 100 MHz.
    - stft_point (int): Number of points for STFT, default is 2048.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.

    Your raw data should organize like this:
    file_path
        Drone 1
            data pack1.dat
            data pack2.dat
            ...
            data packn.dat
        Drone 2
            data pack1.dat
            data pack2.dat
            ...
            data packn.dat
        Drone 3
            data pack1.dat
            data pack2.dat
            ...
            data packn.dat
        .....
        Drone n
            ...
    """
    re_files = os.listdir(file_path)

    for file in re_files:
        packlist = os.listdir(os.path.join(file_path, file))
        for pack in packlist:
            check_folder(os.path.join(os.path.join(fig_save_path + file), pack))
            generate_images(datapack=os.path.join(packlist, pack),
                            file=file,
                            pack=pack,
                            fs=fs,
                            stft_point=stft_point,
                            duration_time=duration_time,
                            ratio=0,
                            location=fig_save_path,
                            )

            print(pack + ' Done')
        print(file + ' Done')
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


def STFT(data,
         onside: bool = True,
         stft_point: int = 1024,
         fs: int = 100e6,
         duration_time: float = 0.1,
         ):

    """
    Performs Short-Time Fourier Transform (STFT) on the given data.

    Parameters:
    - data (array-like): Input data.
    - onside (bool): Whether to return one-sided or two-sided STFT, default is True.
    - stft_point (int): Number of points for STFT, default is 1024.
    - fs (int): Sampling frequency, default is 100 MHz.
    - duration_time (float): Duration time for each segment, default is 0.1 seconds.

    Returns:
    - f (array): Frequencies.
    - t (array): Times.
    - Zxx (array): STFT result.
    """

    slice_point = int(fs * duration_time)

    f, t, Zxx = stft(data[0: slice_point], fs,
         return_onesided=onside, window=windows.hamming(stft_point), nperseg=stft_point)

    return f, t, Zxx


# Usage----------------------------------------------------------------------------------------
def main():
    test = RawDataProcessor()
    """
    test.ShowSpectrogram(data_path='E:/Drone_dataset/RFUAV/crop_data/DJFPVCOMBO/DJFPVCOMBO-16db-90db_5760m_100m_10m/DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat',
                         drone_name='DJ FPV COMBO',
                         sample_rate=100e6,
                         stft_point=2048,
                         duration_time=0.1,
                         Middle_Frequency=2400e6
                         )
    """

    test.TransRawDataintoSpectrogram(fig_save_path='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/images/Py/',
                                     data_path='//UGREEN-8880/zstu320_320_公共空间/RFUAV/加噪/',
                                     sample_rate=100e6,
                                     stft_point=2048,
                                     duration_time=0.1,
                                     )

    """
    
    datapack = 'E:/Drone_dataset/RFUAV/crop_data/DJFPVCOMBO/DJFPVCOMBO-16db-90db_5760m_100m_10m/DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat'
    save_path = 'E:/Drone_dataset/RFUAV/darw_test/'
    save_as_video(datapack=datapack,
                  save_path=save_path,
                  fs=100e6,
                  stft_point=1024,
                  duration_time=0.1,
                  fps=5,
                  )

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
    """

if __name__ == '__main__':
    main()