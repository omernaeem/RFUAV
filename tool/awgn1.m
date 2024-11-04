%  Load data
function [noisy_data] = awgn1(dataIQ,targetSNR_dB);
    % path = "E:\DataBase\DJFPVCOMBO-22db-90db_5760m_100m_40m\DJFPVCOMBO-22db-90db_5760m_100m_40m_0-2s.dat";    % 文件路径
    % data_len = 1.5e6;
    % NFFT = 204800;
    % fs = 100e6;
    % fp = fopen(fullfile(path), "rb");
    % data = fread(fp, data_len * 2, "float");
    % dataIQ = data(1:2:end) + data(2:2:end) * 1j;
    % fclose(fp);
    % dataIQ = normalize(dataIQ, "norm");
    % dataFFT = abs(fftshift(fft(dataIQ, NFFT) / NFFT));
    % dataIQ = dataIQ * 1000;
    % 设定目标 SNR（信噪比）
    % targetSNR_dB = 200; % 目标信噪比，单位为 dB
    targetSNR = 10^(targetSNR_dB / 10); % 将 dB 转换为线性比例
    signalPower = 27;
    % 计算所需的噪声功率
    noisePower = signalPower / targetSNR; % 计算噪声功率
    
    % 生成复数白噪声
    noise = sqrt(noisePower/2) .* (randn(size(dataIQ)) + 1j * randn(size(dataIQ))); % 生成复数白噪声
    noisy_data = dataIQ + noise;
%     figure
%     subplot(311)
%     [pxx, f] = pwelch(noise, hamming(round(length(noise) / 10)), [], 2048000, 100e6, "centered");
%     plot(f, db(pxx))                % 噪声功率谱
%     title("噪声的PSD")
%     subplot(312)
%     [pxx, f] = pwelch(dataIQ, hamming(round(length(dataIQ) / 10)), [], 2048000, 100e6, "centered");
%     plot(f, db(pxx))                % 信号功率谱
%     title("信号的PSD")
    % % awgn 添加噪声
% noisy_data = awgn(dataIQ, targetSNR_dB, 'measured');
% 
% % 绘制结果
% figure;
% subplot(2, 1, 1);
% plot(real(dataIQ), 'b'); hold on;
% plot(imag(dataIQ), 'r');
% title('原始 IQ 信号');
% xlabel('时间 (s)');
% ylabel('幅度');
% legend('实部', '虚部');
% 
% subplot(2, 1, 2);
% plot(real(noisy_data), 'b'); hold on;
% plot(imag(noisy_data), 'r');
% title('添加噪声后的 IQ 信号');
% xlabel('时间 (s)');
% ylabel('幅度');
% legend('实部', '虚部');
% 
% file = "E:\DataBase\DJFPVCOMBO-22db-90db_5760m_100m_40m\DJFPVCOMBO-22db-90db_5760m_100m_40m_0-2s-noise.dat";
% fp = fopen(file,"wb+");
% data(1:2:end) = real(noisy_data);
% data(2:2:end) = imag(noisy_data);
% fwrite(fp,data,"float");
% fclose(fp);
