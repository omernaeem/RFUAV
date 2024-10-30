clc;
clear;
close all;

%% Parameter Init
NFFT = 204800;
data_len = 1.5e6;    % 读取点数
ddc_decifactor = 4; % 下变频降采样指数
fs = 100e6;

%% 下面三个是一定要修改的参数：符号贷款、文件名字、文件路径
band_width = 10e6;  % 信号带宽
path = "G:\1_data_sample\Drones\DJFPVCOMBO\DJFPVCOMBO-16db-90db_5760m_100m_10m";    % 文件路径
file = "DJFPVCOMBO-16db-90db_5760m_100m_10m_0-2s.dat";                              % 文件名字

%% Load data
fp = fopen(fullfile(path, file), "rb");
data = fread(fp, data_len * 2, "float");
dataIQ = data(1:2:end) + data(2:2:end) * 1j;
fclose(fp);
dataIQ = normalize(dataIQ, "norm");

%% 确定频偏与降采样
subplot(311)
[pxx, f] = pwelch(dataIQ, hamming(round(length(dataIQ) / 10)), [], NFFT, fs, "centered");
plot(f, db(pxx))                % 信号功率谱
title("原始信号的PSD")
[~, max_index] = max(db(pxx(500:end-500)));
freq_shift = f(max_index+500);    % 频偏


%% 下变频
tvec = (1:length(dataIQ))' / fs;
carreir = exp(1j*2*pi*-freq_shift.*tvec);
data_baseband = dataIQ .* carreir;
data_resmaple = resample(data_baseband, fs/2, fs);
fs = fs/2;


%% 绘图
subplot(312)
pwelch(data_resmaple, hamming(round(length(data_resmaple) / 10)), [], NFFT, fs, "centered");
title("下变频-降采样处理后信号功率谱")


subplot(313)
temp = abs(fftshift(fft(data_resmaple, NFFT) / NFFT));
plot((temp));
axis tight

index1 = round(NFFT/2 - band_width * 0.75 / 2 / fs * NFFT);
index2 = round(NFFT/2 + band_width * 0.75 / 2 / fs * NFFT);
xline(index1, "r");
xline(index2, "r");
signal_power = mean(temp(index1 : index2) .^ 2);     % 信号功率

index11 = round(NFFT/2 - (band_width * 1.1 + 2.5e6) / 2 / fs * NFFT);
index22 = round(NFFT/2 - band_width  * 1.0 / 2 / fs * NFFT);

xline(index11, "k");
xline(index22, "k");
title("红色区域内为截取的信号，黑色区域内为截取的噪声")
noise_pwoer = mean(temp(index11 : index22) .^ 2);    % 噪声功率


%% SNR Esti
snr_esti = 10*log10((signal_power - noise_pwoer) / noise_pwoer);
fprintf("SNR估计值为: %.2fdb\n", snr_esti);