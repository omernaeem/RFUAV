import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.signal import stft
from graphic import iqreader


def generate_frequency_hopping_signal(frequencies, hop_duration, sample_rate, total_duration):
    t = np.arange(0, total_duration, 1 / sample_rate)
    signal = np.zeros_like(t)

    num_hops = int(total_duration / hop_duration)
    for i in range(num_hops):
        start_idx = int(i * hop_duration * sample_rate)
        end_idx = int((i + 1) * hop_duration * sample_rate)
        freq = np.random.choice(frequencies)
        signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])

    return t, signal


def add_awgn_noise(signal, snr):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal


def psd_snr(datapack, noise, fs, nperseg=1024):

    # 计算信号的功率谱密度
    f_signal, psd_signal = welch(datapack, fs=fs, nperseg=nperseg)

    # 计算噪声的功率谱密度
    f_noise, psd_noise = welch(noise, fs=fs, nperseg=nperseg)

    # 计算信号的平均功率
    signal_power = np.mean(psd_signal)

    # 计算噪声的平均功率
    noise_power = np.mean(psd_noise)

    # 计算 SNR (信号功率 / 噪声功率)
    snr = 10 * np.log10(signal_power / noise_power)
    print(snr)

    return snr


def main():
    signal = 'E:/Drone_dataset/RFUAV/rawdata/crop_data/DJFPVCOMBO/DJFPVCOMBO-16db-90db_5760m_100m_10m/DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat'
    background = "E:/Drone_dataset/RFUAV/rawdata/DJMINI3/KONG-60db_2470m_100m/KONG-60db_2470m_100m_0-2s.dat"
    # 示例用法
    fs = 100e6  # 采样频率，单位 Hz
    signal_data = np.fromfile(signal, dtype=np.float32)
    noise_data =np.fromfile(background, dtype=np.float32)

    snr_value = psd_snr(signal_data, noise_data, fs)
    print("SNR:", snr_value, "dB")


if __name__ == '__main__':
    main()