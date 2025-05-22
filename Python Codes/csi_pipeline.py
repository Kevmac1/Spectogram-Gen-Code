"""
CSI Processing Pipeline Module
"""

import numpy as np
from scipy.signal import butter, lfilter, stft, find_peaks
from scipy.signal.windows import gaussian
from statsmodels.tsa.stattools import acf

FC = 5.18e9
LAMBDA = 3e8 / (2 * FC)  # = 3e8/(2*fc)

def percent(cur, tot):
    return (cur / tot) * 100

def remove_low_freq(x, w):
    k = np.ones(w) / w
    baseline = np.apply_along_axis(lambda c: np.convolve(c, k, mode='same'),
                                   axis=0, arr=x)
    return x - baseline

def remove_noise_floor(S, freq, time):
    mf = freq >= 70
    mt = time > 5
    NF = np.mean(S[np.ix_(mf, mt)])
    S2 = S - NF
    S2[S2 <= 0] = 0
    return S2

def PCA_filter(x, ts, n_pc):
    wlen = int(np.ceil(1/ts))
    nw = x.shape[0] // wlen
    out = []
    for i in range(nw):
        seg = x[i*wlen:(i+1)*wlen, :].astype(np.float64)
        seg = np.nan_to_num(seg)
        if seg.size == 0:
            continue
        try:
            U, S, Vh = np.linalg.svd(seg, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        comps = U[:, :n_pc] * S[:n_pc]
        out.append(comps)
    return np.vstack(out) if out else np.zeros((0, n_pc))

def gaussspec(csi, fs, tmax, tmin):
    ts = 1/fs
    n_win = int(np.floor(0.4/ts))
    n_shift = 8
    f_cut, nPCA = 100, 15

    mag = np.abs(csi).astype(np.float64)
    t = np.arange(mag.shape[0]) * ts
    if tmax != 0:
        mask = (t>=tmin)&(t<=tmax)
        mag = mag[mask, :]
        t = np.arange(mag.shape[0]) * ts

    b, a = butter(10, f_cut/(fs/2))
    rec = lfilter(b, a, mag, axis=0)
    rec = remove_low_freq(rec, int(np.floor(0.3/ts)))
    rec_pca = PCA_filter(rec, ts, nPCA)

    specs = []
    for p in range(nPCA):
        alpha = 2.5
        sigma = (n_win - 1) / (2*alpha)
        win = gaussian(n_win, std=sigma)

        f, times, Z = stft(
            rec_pca[:, p],
            fs=fs,
            window=win,
            nperseg=n_win,
            noverlap=n_win-n_shift,
            nfft=256,               # MATLAB uses 256-point FFT
            boundary='zeros',
            padded=True,
            return_onesided=True
        )
        mf = f <= f_cut
        S = np.abs(Z[mf, :])
        f2 = f[mf]
        S = S / np.sum(S, axis=0, keepdims=True)
        S = remove_noise_floor(S, f2, times)
        specs.append(S)

    spec = np.sum(np.stack(specs, axis=2), axis=2)
    spec = spec / np.sum(spec, axis=0, keepdims=True)
    return times, f2, spec

def extract_torso_speed(freq, spec, pct=45, e_min=0.013): #tweak emin (0.0001)
    nf, nt = spec.shape
    # match MATLAB starting_idx = ceil(nf*0.15) (1-based) → zero-based index = that-1
    start = int(np.ceil(nf * 0.15)) - 1 # tweak where it starts from the bottom of spectrogram (0.15)
    maxf = np.sum(spec[start:, :]**2, axis=0)
    p = np.zeros(nt)
    for ti in range(nt):
        m = start
        en = np.sum(spec[start:m+1, ti]**2)
        pr = percent(en, maxf[ti])
        while pr < pct and m < nf - 1:
            m += 1
            en = np.sum(spec[start:m+1, ti]**2)
            pr = percent(en, maxf[ti])
        p[ti] = freq[m] if maxf[ti] > e_min else 0.0
    return p * LAMBDA

def find_stable_section(p, w=175, s=10, vmin=0.8):
    vars_, avgs, starts = [], [], []
    for st in range(0, len(p) - w + 1, s):
        seg = p[st:st + w]
        av = np.mean(seg)
        if av < vmin:
            continue
        # MATLAB var uses ddof=1
        vars_.append(np.var(seg, ddof=1))
        avgs.append(av)
        starts.append(st)
    if not vars_:
        return None
    idx = int(np.argmin(vars_))
    return {
        'start': starts[idx],
        'end':   starts[idx] + w - 1,
        'avg_velocity': avgs[idx],
        'variance': vars_[idx]
    }

def compute_autocorr_features(spec, si, ei):
    nf = spec.shape[0]
    tr = slice(si, ei + 1)
    acm = np.array([acf(spec[i, tr], nlags=2, fft=False)
                    for i in range(nf)])
    energy = np.sum(spec[:, tr], axis=1)
    energy /= np.max(energy)
    acv = np.sum(acm[:, 1:3], axis=1) * energy
    auto_corr = np.sum(acv)
    grad_ac = np.gradient(acv)
    return auto_corr, np.mean(grad_ac), np.std(grad_ac), grad_ac

def detect_gait(seg, tseg):
    prom = 0.1 * (seg.max() - seg.min())
    dist = 0.3 / np.mean(np.diff(tseg))
    peaks, _ = find_peaks(seg, prominence=prom, distance=dist)
    if len(peaks) >= 2:
        cycles = np.diff(tseg[peaks])
        return np.mean(cycles), peaks
    return np.nan, None
