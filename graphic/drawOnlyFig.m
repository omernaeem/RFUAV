%% 多种频率分辨率以及颜色映射画图
clc;clear;close all;
fs = 100e6;                     % 输入采样率
fftpoint = [128,256,512,1024];
time_sec = 0.1;                   % 需要的分割时间/s
dataform = 'float32';           % 输入的数据类型
byte_per = 4;                   % 该数据类型占字节数
datalength = time_sec*fs*byte_per*2;       % 读取数据的长度，单位是字节(时间*采样率*每个数据占字节*iq)
file_in ="Z:\RFUAV\加噪\DJMAVIC3PRO-16db-90db_5800m_100m_20-noise";% 输入路径
% 获取文件夹内所有iq/dat文件
files_dat = dir(fullfile(file_in, '*.dat'));
files_iq = dir(fullfile(file_in, '*.iq'));
files = [files_dat; files_iq];
% 一级循环，遍历文件
for ii = 1:length(files)
    fileName{ii} = files(ii).name;
    file_input = fullfile(file_in,fileName{ii});
    
    myname = char(file_in);
    for i = length(myname):-1:1
        if strcmp(myname(i), '\')
            path = string(myname(i+1:length(myname)));
            break; % 找到第一个 '-' 后退出循环
        end
    end
%     filepathOut = "E:\Drone_dataset\RFUAV\augmentation_exp1_MethodSelect\images\Matlab";
    filepathOut = "E:\DataBase\stftFig";
    filepathOut_get = filepathOut + '\' + path + '\' + fileName{ii}(1:end-4);
    color = ["parula","hsv","hot","autumn"];
    % 读取文件,获取大小
    fp = fopen(file_input, 'rb'); 
    fseek(fp, 0, 1);
    fileSize = ftell(fp);
    fclose(fp);
    readtime = ceil(fileSize/datalength);
    
    %% 分次读取文件保存
    time = 0;
    for i =1:readtime
        tic
        fp = fopen(file_input, 'rb'); 
        fseek(fp,(i-1)*datalength,-1);
        data = fread(fp,datalength/4,dataform);
        fclose(fp);
        dataIQ = data(1:2:end-1) + 1i * data(2:2:end);
        clear data;
        for j = 1:length(fftpoint)
            for k = 1:length(color)
                stft(dataIQ,fs,FFTLength=fftpoint(j));
                filepathOut = filepathOut_get + '\'+ color(k) + '\'+num2str(fftpoint(j));
                if ~exist(filepathOut,"dir")
                    mkdir(filepathOut);
                end
                newFile = fullfile(filepathOut,num2str(i*0.1-0.1+time) + "-" + num2str(i*0.1+time) + "s-" +...
                    color(k) + "-" + num2str(fftpoint(j)) + ".jpg"); % 生成新的文件路径和文件名
                % 移除坐标轴和标题等内容
                axis off; % 隐藏坐标轴
                set(gca, 'Position', [0 0 1 1]); % 将轴的大小扩展到整个图像区域
                
                % 保存图像为 2400x1800 像素
                set(gcf, 'Units', 'inches', 'Position', [0, 0, 4, 3]); % 调整图像窗口大小
                print(gcf, newFile, '-dpng', '-r300');
                toc
                clf;
            end
        end
        toc
    end
end