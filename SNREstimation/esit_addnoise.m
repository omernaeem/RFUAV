clc;
clear;
close all;

%% Parameter Init
NFFT = 409600;
data_len = 1.5e6;       % 读取点数
ddc_decifactor = 4;     % 下变频降采样指数
fs = 100e6;
snrRange = [-2,-20];   % 目标信噪比范围

%% 下面三个是一定要修改的参数：符号带宽、文件名字、文件路径
bw = 20e6;      % 信号带宽
% 获取文件夹内所有iq、dat文件
file_in = "E:\DataBase\加噪\20M";
file_out = "Z:\RFUAV\加噪";
files_dat = dir(fullfile(file_in, '*.dat'));
files_iq = dir(fullfile(file_in, '*.iq'));
files = [files_dat; files_iq];

for ii = length(files):-1:1
    fileName{ii} = files(ii).name;
    % Load data
    fp = fopen(fullfile(file_in,fileName{ii}), "rb");
    data = fread(fp, data_len*2,"float");
    dataIQ = data(1:2:end) + data(2:2:end) * 1j;
    fclose(fp);
%     dataIQ = normalize(dataIQ, "norm");
    % 信噪比估计

    % 加噪
    [idx1,idx2,idx3,idx4,f1, f2] = positionFind(dataIQ, fs, bw, NFFT);
    snr_esti = snrEsti(dataIQ,fs,NFFT,f1,f2,idx1,idx2,idx3,idx4);
    noise = snr_esti-15;
    rawSnr = snr_esti;
    for SNR = snrRange(1):-2:snrRange(2)
        while(abs(snr_esti - SNR) > 0.1)
%             if(abs(snr_esti - SNR) > 2)
%                 add = 2;
%             else
%                 add = 0.1;
%             end
            add = abs(snr_esti - SNR) / 2;
            if(snr_esti - SNR>0)
                noise = noise - add;
            else
                noise = noise + add;
            end
%             snr_esti = snrEsti(dataIQ,fs,bw, NFFT)
            datanoise = awgn1(dataIQ,noise,rawSnr);

            % 计算加噪后信噪比
            snr_esti = snrEsti(datanoise,fs,NFFT,f1,f2,idx1,idx2,idx3,idx4);
            disp("信噪比"+snr_esti);
        end
        % 保存文件
        DoAddNoise(file_out,file_in,fileName{ii},rawSnr,noise,SNR);
    end
end