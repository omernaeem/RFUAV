clc;
clear;
close all;

<<<<<<< Updated upstream
% 参数设置
fs = 100e6;                   % 采样率
fc = 2440e6;                  % 中心频率
fftlength = [2048,1024, 512, 128]; % 不同的FFT长度
time_sec = 0.5;                   % 需要的分割时间/s
dataform = 'float32';           % 输入的数据类型
byte_per = 4;                   % 该数据类型占字节数
datalength = time_sec*fs*byte_per*2;
file = "Z:\RFUAV\UAVDATA\DJMINI4PRO\DIMINI4PRO-17db-60db_2450m_100m_20m.iq"; % 文件路径

ii = 2;
blocksize = 1e6;            % 每次读取的数据大小
times = datalength / blocksize; % 总数据块数
overlap = 0.25;
outLengthPer = 1+floor((blocksize/2 - fftlength(ii)) / (fftlength(ii)*(1-overlap)));
sTotal = zeros(fftlength(ii),times*outLengthPer);                % 用于存储完整 STFT 矩阵
tTotal = zeros(1,times*outLengthPer);                % 存储全局时间
% 循环处理每个数据块
=======
% args
fs = 100e6;                   % sample rate
fc = 2440e6;                  
fftlength = [2048,1024, 512, 128]; 
time_sec = 0.5;                   
dataform = 'float32';           
byte_per = 4;                   
datalength = time_sec*fs*byte_per*2;
file = ""; % path

ii = 2;
blocksize = 1e6;            
times = datalength / blocksize; 
overlap = 0.25;
outLengthPer = 1+floor((blocksize/2 - fftlength(ii)) / (fftlength(ii)*(1-overlap)));
sTotal = zeros(fftlength(ii),times*outLengthPer);
tTotal = zeros(1,times*outLengthPer);

>>>>>>> Stashed changes
for i = 1:times
    fp = fopen(file, "rb");
    fseek(fp,blocksize*(i-1),-1);
    data = fread(fp, blocksize, dataform);
    fclose(fp);
    dataIQ = data(1:2:end) + 1j * data(2:2:end);  
    clear data;
    [s, f, t] = spectrogram(dataIQ,fftlength(ii),fftlength(ii)*overlap,fftlength(ii),fs);
    sTotal = [sTotal, s];
    tTotal = [tTotal, t + (i-1)*blocksize/fs];
<<<<<<< Updated upstream
    
% 
%     surf(tTotal, f, abs(sTotal), 'EdgeColor', 'none'); % 绘制频谱图（幅度）
%     axis xy;  % Y轴方向正向
%     colormap jet;  % 颜色映射
% %     caxis([min(min(abs(sTotal))), max(max(abs(sTotal)))]);
%     xlabel('Time (s)');
%     ylabel('Frequency (Hz)');
%     title('STFT of IQ Signal');
%     
%     % 设置视角
%     view(-45, 30);  % 设置3D视角
%     colorbar;  % 显示颜色条    
end
clear dataIQ;
% 画出STFT结果
figure;
surf(tTotal, f, abs(sTotal), 'EdgeColor', 'none'); % 绘制频谱图（幅度）
axis xy;  % Y轴方向正向
colormap jet;  % 颜色映射
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT of IQ Signal');
% xlim([0, 0.4]);
% ylim([0, 10]);
% var = {'s','sTotal','tTotal'};
% clear (var{:});
% 设置视角
view(-45, 60);  % 设置3D视角
colorbar;  % 显示颜色条
=======
      
end
clear dataIQ;

figure;
surf(tTotal, f, abs(sTotal), 'EdgeColor', 'none');
axis xy;
colormap jet;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT of IQ Signal');
colorbar;

% Setting the Viewing Angle
view(-45, 60);
>>>>>>> Stashed changes
