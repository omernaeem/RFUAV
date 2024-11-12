function [outputArg1,outputArg2] = DoAddNoise(file_in,fileName,signal_power,noisevalue,target)
%   此处显示详细说明
    targetSNR = 10^(noisevalue / 10); % 将 dB 转换为线性比例
    % 计算所需的噪声功率
    noisePower = signal_power / targetSNR; % 计算噪声功率
    
    file = fullfile(file_in,fileName);
    fileInfo = dir(file); % 替换为你的文件名
    fileSize = fileInfo.bytes; % 获取文件大小，单位为字节
    times = fileSize /4/ 3e6;  
    filepathOut = file_in + '\' + fileName(1:end-4) + '-noise' ;
    if ~exist(filepathOut,"dir")
        mkdir(filepathOut);
    end
    filepathOut = filepathOut + '\' + num2str(target)+ "dB.dat";  
    for i = 1:times
        fp = fopen(file,"rb"); 
        fseek(fp,3e6*(i-1),-1);
        data = fread(fp,3e6,"float");
        fclose(fp);
        dataIQ = data(1:2:end) + 1j*data(2:2:end);
        % 生成复数白噪声
        noise = sqrt(noisePower/2) .* (randn(size(dataIQ)) + 1j * randn(size(dataIQ))); % 生成复数白噪声
        noisy_data = dataIQ + noise;
        
%     datanoise = awgn1(dataIQ,noise);
        % 保存  
        if(i == 1)
            fp = fopen(filepathOut,"wb+");
        else
            fp = fopen(filepathOut,"ab+");
        end
        data=zeros(length(dataIQ)*2,1);
        data(1:2:end) = real(noisy_data);
        data(2:2:end) = imag(noisy_data);
        fwrite(fp,data,"float");
        fclose(fp);
    end
end

