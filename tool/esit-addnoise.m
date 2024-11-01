clc;
clear;
close all;

%% Parameter Init
NFFT = 204800;
data_len = 1.5e6;    % 读取点数
ddc_decifactor = 4; % 下变频降采样指数
fs = 100e6;

%% 下面三个是一定要修改的参数：符号贷款、文件名字、文件路径
band_width = 40e6;  % 信号带宽
% 获取文件夹内所有iq文件
file_in = "E:\DataBase\DJFPVCOMBO-22db-90db_5760m_100m_40m";
files = dir(fullfile(file_in,'*.dat'));
for ii = length(files)-2:-1:2
    fileName{ii} = files(ii).name;
    file_input = fullfile(file_in,fileName{ii});
    % Load data
    fp = fopen(file_input, "rb");
    data = fread(fp, data_len*2,"float");
    dataIQ = data(1:2:end) + data(2:2:end) * 1j;
    fclose(fp);

    fp = fopen(file_input, "rb");
    dataall = fread(fp, "float");
    dataIQall = dataall(1:2:end) + dataall(2:2:end) * 1j;
    fclose(fp);
    % dataIQ1 = normalize(dataIQ, "norm");
    % 加噪
    % noise = 68.7---20dB加噪
    % noise = 66.5---20dB
    % noise = 64.3---16dB
    % noise = 62.1---14dB
    % noise = 60  ---12dB
    % noise = 58  ---10dB
    % noise = 56  ---8 dB
    % noise = 54  ---6 dB
    snr_esti = 100;
    SNR = 20;
    noise = SNR + 48;
    for SNR = 20:-2:0
        while(abs(snr_esti - SNR) > 0.05)
            if(snr_esti - SNR>0)
                noise = noise - 0.1
            else
                noise = noise + 0.1
            end
            datanoise = awgn1(dataIQ,noise);
            dataIQ1 = datanoise;            
            % 确定频偏与降采样
            NFFT = 204800;
            data_len = 1.5e6;    % 读取点数
            ddc_decifactor = 4; % 下变频降采样指数
            fs = 100e6;
            [pxx, f] = pwelch(dataIQ1, hamming(round(length(dataIQ1) / 10)), [], NFFT, fs, "centered");
            [~, max_index] = max(db(pxx(500:end-500)));
            freq_shift = f(max_index+500);    % 频偏
            % 下变频
            tvec = (1:length(dataIQ1))' / fs;
            carreir = exp(1j*2*pi*-freq_shift.*tvec);
            data_baseband = dataIQ1 .* carreir;
            data_resmaple = resample(data_baseband, fs/2, fs);
            fs = fs/2;
            temp = abs(fftshift(fft(data_resmaple, NFFT) / NFFT));
            
            index1 = round(NFFT/2 - band_width * 0.75 / 2 / fs * NFFT);
            index2 = round(NFFT/2 + band_width * 0.75 / 2 / fs * NFFT);
            signal_power = mean(temp(index1 : index2) .^ 2);     % 信号功率
            
            index11 = round(NFFT/2 - (band_width * 1.1 + 2.5e6) / 2 / fs * NFFT);
            index22 = round(NFFT/2 - band_width  * 1.0 / 2 / fs * NFFT);
            noise_pwoer = mean(temp(index11 : index22) .^ 2);    % 噪声功率
            % SNR Esti
            snr_esti = 10*log10((signal_power - noise_pwoer) / noise_pwoer);
            fprintf("SNR估计值为: %.2fdb\n", snr_esti);
        end
        % 保存文件
        DoAddNoise(file_in,fileName{ii},dataIQall,signal_power,noise,SNR);
    end
end