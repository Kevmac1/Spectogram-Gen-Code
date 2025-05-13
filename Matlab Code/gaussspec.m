%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Spectrom Generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [time, freq, spectrogram]=gaussspec(csi,fs,tmax, tmin)
fc = 5.18e9; %carrier frequency
lambda = 3e8/fc; %wavelength
ts = 1/fs; %sampling time
time_window = 0.4; n_win=floor(time_window/ts); 
n_shift=8;%shift between time windows
fmax=100;
freq = 1:fmax; % frequencies in the spectrogram
nPCA = 15; %number of PCA components
% add=extractbetween()
wifi_csi=abs(csi);
t = ts*(0:size(wifi_csi,1)-1);         
% Butterworth LPF 
f_cut = 100;
% one section walking
if tmax~=0
    wifi_csi=wifi_csi(find(t>=tmin,1):find(t<=tmax,1),:);
    n=size(wifi_csi,1);
    t=0:ts:(n-1)*ts;
end
[b,a] = butter(10,f_cut/(fs/2));
rec_sig_butter = filter(b,a,wifi_csi);
rec_sig_butter = remove_low_freq(rec_sig_butter, floor(.3/ts));
rec_sig_PCA = PCA_filter(rec_sig_butter, ts, (1:nPCA));

% Generate the spectrogram of each PCA component
for p = 1:nPCA
    [S,freq,time]=stft(rec_sig_PCA(:,p),fs,"Window",gausswin(n_win),"overlapLength",n_win-n_shift,"FrequencyRange","onesided");   
    S=S(freq<=fmax,:);
    freq=freq(freq<=fmax);
    S = abs(S);
    S = S./sum(S,1);
    S = remove_noise_floor(S, freq,time);
    spectrogram(:,:,p) = abs(S);
end
%add the spectrograms of all PCA components
spectrogram = sum(abs(spectrogram),3);

%normalize such that sum of each column is 1
spectrogram = spectrogram ./sum(spectrogram,1);
end