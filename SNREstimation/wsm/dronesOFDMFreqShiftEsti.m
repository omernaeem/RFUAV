function [f1, f2] = dronesOFDMFreqShiftEsti(x, fs, bw, nfft)
% dronesOFDMbandwidthEsti: 无人机OFDM信号频偏粗检测
% x:  input signal
% fs: sample rate
% bw: bandwitdh 

[pxx, fvec] = pwelch(x, hamming(round(length(x)/ 10)), [], nfft, fs, "centered");
pxx = db(pxx(1:end));
fvec = fvec(1:end);

pxx = pxx / max(abs(pxx));  % 能量归一化
pxx = pxx + (-min(pxx));    % 置最小值为0


bwNfft = round(nfft * (bw / fs) * 0.9); % 信号带宽对应FFT点数

energy = zeros(nfft - bwNfft - 20, 1);
energy(1) = mean(pxx(1:bwNfft) .^ 2);

for i = 2 : length(energy) 
    energy(i) = energy(i-1) - (pxx(i-1)^2)/bwNfft;
    energy(i) = energy(i)   + (pxx(i+bwNfft-1)^2)/bwNfft;
end
[~, maxIdx] = max(energy);


f1 = fvec(maxIdx);
f2 = fvec(maxIdx + bwNfft);


end

