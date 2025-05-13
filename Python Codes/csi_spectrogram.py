# csi_spectrogram.py

import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from parse_csi import parse_csi_from_raw_csi_file

def remove_low_freq(x, window_len):
    filt = np.ones(window_len) / window_len
    if x.ndim == 1:
        low_freq = np.convolve(x, filt, mode='same')
    else:
        low_freq = signal.convolve2d(x, filt[:, None], mode='same')
    return x - low_freq

def PCA_filter(data, ts, components=15):
    win_len = int(np.ceil(1 / ts))
    num_windows = data.shape[0] // win_len
    filtered = []
    for i in range(num_windows):
        segment = data[i * win_len:(i + 1) * win_len, :]
        pca = PCA()
        proj = pca.fit_transform(segment)
        filtered.append(proj[:, :components])
    return np.vstack(filtered)

def remove_noise_floor(spec, f, t, floor_freq=70, floor_time=5):
    f_idx = f >= floor_freq
    t_idx = t > floor_time
    S = spec[np.ix_(f_idx, t_idx)]
    floor = np.mean(S)
    spec_clean = spec - floor
    spec_clean[spec_clean < 0] = 0
    return spec_clean

def generate_csi_spectrogram(csi_data, fs=500):
    ts = 1 / fs
    f_c = 5.18e9
    lambda_c = 3e8 / f_c

    csi_mag = np.abs(csi_data)
    b, a = signal.butter(10, 100 / (fs / 2), btype='low')
    csi_lp = signal.filtfilt(b, a, csi_mag, axis=0)
    csi_detrended = remove_low_freq(csi_lp, int(np.floor(0.3 / ts)))
    csi_pca = PCA_filter(csi_detrended, ts, components=15)

    win_sec = 0.4
    n_win = int(win_sec / ts)
    n_shift = 8
    f_max = 100
    gauss_win = signal.windows.gaussian(n_win, std=n_win / 6)

    spectrograms = []
    for i in range(csi_pca.shape[1]):
        f, t, Sxx = signal.stft(csi_pca[:, i],
                                fs=fs,
                                window=gauss_win,
                                nperseg=n_win,
                                noverlap=n_win - n_shift,
                                nfft=4 * n_win,
                                boundary=None)
        f_idx = f <= f_max
        Sxx = np.abs(Sxx[f_idx, :])
        f = f[f_idx]
        Sxx /= np.sum(Sxx, axis=0, keepdims=True)
        Sxx = remove_noise_floor(Sxx, f, t)
        spectrograms.append(Sxx)

    spec_final = np.sum(spectrograms, axis=0)
    spec_final /= np.sum(spec_final, axis=0, keepdims=True)
    v_axis = f * lambda_c / 2  # Doppler: v = f * λ / 2

    return t, v_axis, spec_final

def main():
    csi_path = os.path.join(os.path.dirname(__file__), "Fei_walk2.csi") # Change to target file name
    csi = parse_csi_from_raw_csi_file(csi_path)

    t, v, S = generate_csi_spectrogram(csi, fs=500)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, v, S, shading='gouraud')
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.title('CSI Spectrogram with Signal Processing')
    plt.colorbar(label='Normalized Magnitude')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()