%  Load data
function [noisy_data] = awgn1(dataIQ,targetSNR_dB);
    targetSNR = 10^(targetSNR_dB / 10); % 将 dB 转换为线性比例
    signalPower = 27;
    % 计算所需的噪声功率
    noisePower = signalPower / targetSNR; % 计算噪声功率
    
    % 生成复数白噪声
    noise = sqrt(noisePower/2) .* (randn(size(dataIQ)) + 1j * randn(size(dataIQ))); % 生成复数白噪声
    noisy_data = dataIQ + noise;

