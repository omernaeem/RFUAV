%% 依据需要的时间来分割大数据文件
clc;
clear;
close all;

fs = 100e6;                     % 输入采样率
fftpoint = 8192;
time_sec = 0.3;                   % 需要的分割时间/s
dataform = 'float32';           % 输入的数据类型
byte_per = 4;                   % 该数据类型占字节数
datalength = time_sec*fs*byte_per*2;       % 读取数据的长度，单位是字节(时间*采样率*每个数据占字节*iq)
file_input ="E:\Drone_dataset\RFUAV\exp3\rawdata\FLYSKY_FS_I6X.iq";% 输入路径
[filepath, name, ~] = fileparts(file_input); % 获取路径、文件名和后缀名，在原路径创建文件夹存放分割后数据
filepath = filepath + '\' + name;

if ~exist(filepath,"dir")
    mkdir(filepath);
else
    disp("文件夹已经存在!");
end

% 读取文件,获取大小
fp = fopen(file_input, 'rb'); 
fseek(fp, 0, 1);
fileSize = ftell(fp);
fclose(fp);
readtime = ceil(fileSize/datalength);

%% 分次读取文件保存
for i =1:readtime
    tic
    fp = fopen(file_input, 'rb'); 
    fseek(fp,(i-1)*datalength,-1);
    data = fread(fp,datalength/4,dataform);
    fclose(fp);
    newFile = fullfile(filepath, name + '_' + num2str(i*2-2) + '-' + num2str(i*2) + 's.iq'); % 生成新的文件路径和文件名
    fp_write = fopen(newFile,"wb+");
    fwrite(fp_write,data,dataform);
    fclose(fp_write);
    toc
end