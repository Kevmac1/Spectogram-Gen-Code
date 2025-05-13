function spectrogram2 = remove_noise_floor(spectrogram, freq,time)
S = spectrogram(freq>=70, time > 5);
NF = mean(S(:)) ;
spectrogram2 = spectrogram - NF;
spectrogram2(spectrogram <= 0) = 0;
