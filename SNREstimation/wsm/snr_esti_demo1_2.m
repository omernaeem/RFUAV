clc;
clear;
close all;

%% Parameter Init
nfft = 409600;
len  = 1.5e6;
fs   = 100e6;



%% 下面三个是一定要修改的参数：符号带宽、文件名字、文件路径
bw = 40e6;  % 信号带宽

path = "Z:\RFUAV\加噪\DJFPVCOMBO-28db-90db_5760m_100m_40-noise\";        % 文件路径
file = "-20dB.dat";    % 文件名字

% path = "G:\1_data_sample\Drones\DJFPVCOMBO\DJFPVCOMBO-28db-90db_5760m_100m_40m";        % 文件路径
% file = "DJFPVCOMBO-28db-90db_5760m_100m_40m_12-14s.dat";    % 文件名字

%% Load data
fp = fopen(fullfile(path, file), "rb");
data = fread(fp, len * 2, "float");
dataIQ = data(1:2:end) + data(2:2:end) * 1j;
fclose(fp);
dataIQ = normalize(dataIQ, "norm");


%% 信号位置检测
[f1, f2] = dronesOFDMFreqShiftEsti(dataIQ, fs, bw, nfft);
f = (f1 + f2) / 2;
sig = dronesOFDMFreqCompensation(dataIQ, fs, f);
f11 = f1 - f;   f22 = f2 - f;



%% 降采样
refs = fs/2;
reNfft = nfft / 2;
sigResample = resample(sig, refs, fs);

[pxx3, fvec3] = pwelch(sigResample, ...
    hamming(round(length(sigResample)/ 10)), [], reNfft, refs, "centered");


%% SNR Esti
temp11 = abs(fvec3 - f11);
temp22 = abs(fvec3 - f22);
[~, idx1] = min(temp11);
[~, idx2] = min(temp22);


bwNoise = 1e6;       % 取一段带宽为1M的信号作为噪声
f3  = 0.75e6;       

bwNoiseNfft = round(reNfft * (bwNoise / refs));
idx4 = idx1 - round(reNfft * (f3 / refs));
idx3 = idx4 - bwNoiseNfft;


fftTemp = abs(fftshift(fft(sigResample, reNfft) / reNfft));
sigPower = mean(fftTemp(idx1:idx2) .^ 2);
nosPower = mean(fftTemp(idx3:idx4) .^ 2);

snrEsti = 10 * log10((sigPower-nosPower)/nosPower);
fprintf("The estimation value of Snr is: %.2fdb\n", snrEsti);


%% Figure
[pxx, fvec] = pwelch(dataIQ, hamming(round(length(dataIQ)/ 10)), [], nfft, fs, "centered");
pxx = db(pxx);
subplot(311)
plot(fvec, pxx);    xline(f1, 'r');  xline(f2, 'r');
title("原始信号")

subplot(312)
[pxx2, fvec2] = pwelch(sig, hamming(round(length(sig)/ 10)), [], nfft, fs, "centered");
pxx2 = db(pxx2);
plot(fvec2, pxx2);    xline(f11, 'r');  xline(f22, 'r');
title("信号：载波矫正")

subplot(313)
pxx3 = db(pxx3);
plot(fvec3, pxx3);    
xline(fvec3(idx1), 'r');  xline(fvec3(idx2), 'r');
xline(fvec3(idx3), 'k');  xline(fvec3(idx4), 'k');
title("信号：降采样")
