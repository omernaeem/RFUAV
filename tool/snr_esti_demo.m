clc;
clear;
close all;

%% Parameter Init
NFFT = 40960;
data_len = 1e6;    % 读取点数
ddc_decifactor = 2; % 下变频降采样指数
fs = 100e6;
band_width = 10e6;  % 信号带宽：40M、20M、10M。该函数与下变频的滤波器有关

%% Load data
path = "E:\Drone_dataset\RFUAV\rawdata\crop_data\DJFPVCOMBO\DJFPVCOMBO-16db-90db_5760m_100m_10m";          % 文件名字
file = "DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat";          % 文件路径
fp = fopen(fullfile(path, file), "rb");
data = fread(fp, data_len * 2, "float");
dataIQ = data(1:2:end) + data(2:2:end) * 1j;
fclose(fp);


%% DDC
% DDC，全名为 Digital Down Converter，数字下变频
% 由于数据集采集到的信号存在频偏，在做SNR估计前需对

subplot(311)
[pxx, f] = pwelch(dataIQ, hamming(round(length(dataIQ) / 10)), [], NFFT, fs, "centered");
plot(f, db(pxx))                % 信号功率谱
title("原始信号的PSD")
[~, max_index] = max(db(pxx(500:end-500)));
freq_shift = f(max_index+500);    % 频偏

ddc_object = dsp.DigitalDownConverter("SampleRate", fs, ...
    "DecimationFactor", ddc_decifactor, ...
    "StopbandAttenuation", 60, ...
    "CenterFrequency", freq_shift, ...
    "PassbandRipple", 0.02, ...
    "Bandwidth", band_width * 1.25);
dataIQ_ddc = ddc_object(dataIQ);
fs = fs / ddc_decifactor;

subplot(312)
pwelch(dataIQ_ddc, hamming(round(length(dataIQ_ddc) / 10)), [], NFFT, fs, "centered");
title("下变频-降采样处理后信号功率谱")

[psd2, f2] = pwelch(dataIQ_ddc, hamming(round(length(dataIQ_ddc) / 10)), [], NFFT, fs, "centered");
subplot(313)
temp = abs(fftshift(fft(dataIQ_ddc, NFFT) / NFFT));
plot(temp);
axis tight

index1 = round(NFFT/2 - band_width * 0.75 / 2 / fs * NFFT);
index2 = round(NFFT/2 + band_width * 0.75 / 2 / fs * NFFT);
xline(index1, "r");
xline(index2, "r");
signal_power = mean(temp(index1 : index2) .^ 2);     % 信号功率

index11 = round(NFFT/2 - (band_width * 1.1 + 5e6) / 2 / fs * NFFT);
index22 = round(NFFT/2 - band_width  * 1.1 / 2 / fs * NFFT);

xline(index11, "k");
xline(index22, "k");
title("红色区域内为截取的信号，黑色区域内为截取的噪声")
noise_pwoer = mean(temp(index11 : index22) .^ 2);    % 噪声功率

%% SNR Esti
snr_esti = 10*log10((signal_power - noise_pwoer) / noise_pwoer);
fprintf("SNR估计值为: %2fdb\n", snr_esti);