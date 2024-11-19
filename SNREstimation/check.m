% 检测噪声效果
clc;clear;
filein = "E:\DataBase\DJFPVCOMBO-22db-90db_5760m_100m_40m\DJFPVCOMBO-22db-90db_5760m_100m_40m_0-2s-noise";
files = dir(fullfile(filein,'*.dat'));
% 按时间排序
[~, idx] = sort([files.datenum]); % 获取文件的修改时间
sortedFiles = files(idx); % 按时间排序的文件列表
figure
nfft = 4096;
for ii = 1:length(files)
    fileName{ii} = sortedFiles(ii).name;
    file_input = fullfile(filein,fileName{ii});
    subplot(3,4,ii)
    fp = fopen(file_input,'rb');
    data = fread(fp,10e6,"float");
    fclose(fp);
    dataIQ = data(1:2:end) +  1j*data(2:2:end);
    stft(dataIQ,100e6);
    title(fileName{ii});
end