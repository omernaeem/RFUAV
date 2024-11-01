function [outputArg1,outputArg2] = DoAddNoise(file_in,fileName,dataIQ,signal_power,noisevalue,target)
%   此处显示详细说明
    targetSNR = 10^(noisevalue / 10); % 将 dB 转换为线性比例
    % 计算所需的噪声功率
    noisePower = signal_power / targetSNR; % 计算噪声功率
    
    % 生成复数白噪声
    noise = sqrt(noisePower/2) .* (randn(size(dataIQ)) + 1j * randn(size(dataIQ))); % 生成复数白噪声
    noisy_data = dataIQ + noise;
    
    % 保存
    filepathOut = file_in + '\' + fileName(1:end-4) + '-noise' ;
    if ~exist(filepathOut,"dir")
        mkdir(filepathOut);
    end
    filepathOut = filepathOut + '\' + fileName(1:end-4) + '_' + num2str(target)+ "dB.dat";
    fp = fopen(filepathOut,"wb+");
    data=zeros(length(dataIQ)*2,1);
    data(1:2:end) = real(noisy_data);
    data(2:2:end) = imag(noisy_data);
    fwrite(fp,data,"float");
    fclose(fp);

end

