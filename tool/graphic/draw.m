%% 多种频率分辨率以及颜色映射画图
clc;clear;close all;
fs = 100e6;                     % 输入采样率
fftpoint = [128,256,512,1024];
time_sec = 0.1;                   % 需要的分割时间/s
dataform = 'float32';           % 输入的数据类型
byte_per = 4;                   % 该数据类型占字节数
datalength = time_sec*fs*byte_per*2;       % 读取数据的长度，单位是字节(时间*采样率*每个数据占字节*iq)
file_input ="E:\DataBase\DJFPVCOMBO-22db-90db_5760m_100m_40m\DJFPVCOMBO-22db-90db_5760m_100m_40m_2-4s.dat";% 输入路径
[filepath, name, ~] = fileparts(file_input); % 获取路径、文件名和后缀名，在原路径创建文件夹存放分割后数据
myname = char(name);
flytype = string(myname(1:end-30));
filepath = filepath + '\' + name;
color = ["hsv","hot","autumn"];
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
    dataIQ = data(1:2:end-1) + 1i * data(2:2:end);
    fclose(fp);
    time = 2;
    for j = 1:4
        for k = 1:3
            stft(dataIQ,fs,FFTLength=fftpoint(j));
            colormap(color(k));
            yticks([-50 :10:50]);
            yticklabels([5710:10:5810]);
            xticks([0:10:100]);
            xticklabels([0:0.1/10:0.1]);
            xlabel("时间(s)");
            title(flytype);
            newFile = fullfile(filepath,num2str(i*0.1-0.1+time) + "-" + num2str(i*0.1+time) + "s-" +...
                color(k) + "-" + num2str(fftpoint(j)) + ".jpg"); % 生成新的文件路径和文件名
            saveas(gcf,newFile);
        end
    end
    toc
end