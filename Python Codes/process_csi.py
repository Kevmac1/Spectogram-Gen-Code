#!/usr/bin/env python3
"""
Parse .csi, compute spectrogram + features, plot, then exit on Enter.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# Get the path to the input file from the command-line argument
input_path = sys.argv[1]

# Extract base name without extension (e.g., 'Katheryn5')
file_base = os.path.splitext(os.path.basename(input_path))[0]
print(file_base)
plt.ion()

try:
    from picoscenes import Picoscenes
except ImportError:
    Picoscenes = None

from csi_pipeline import (
    gaussspec,
    extract_torso_speed,
    find_stable_section,
    compute_autocorr_features,
    detect_gait
)

def parse_csi(fp):
    if Picoscenes is None:
        raise ImportError("Install PicoScenes toolbox.")
    reader = Picoscenes(fp)
    rows, L, skipped = [], None, 0
    for frm in reader.raw:
        try:
            arr = np.asarray(frm["CSI"]["CSI"])
            if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
                r = np.concatenate((arr[:,0,0], arr[:,0,1]))
            else:
                r = arr.ravel()
        except:
            continue
        if L is None:
            L = r.size; rows.append(r)
        elif r.size == L:
            rows.append(r)
        else:
            skipped += 1
    if not rows:
        raise RuntimeError("No valid CSI frames.")
    print(f"Parsed {len(rows)} frames, skipped {skipped}")
    return np.vstack(rows).astype(np.complex64)

def main():

    print("Current working directory:", os.getcwd())
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/file.csi"); sys.exit(1)
    fp = sys.argv[1]

    # iterate through fp (filepath) to determine if it is a .csi file or a .npy file
    # if .csi, proceed as normal
    if fp.endswith('.csi'):
        print("Processing .csi file")

        if not os.path.isfile(fp):
            print("File not found."); sys.exit(1)

        t0 = time.perf_counter()
        csi = parse_csi(fp)
        print(f"Loaded CSI {csi.shape} in {time.perf_counter() - t0:.2f}s")

        # 1) Full spectrogram
        t, f, S = gaussspec(csi, fs=500, tmax=0, tmin=18)
        vel = f * (3e8/(2*5.18e9))

        plt.figure()
        plt.pcolormesh(t, vel, S, shading='auto')
        plt.title("Spectrogram")
        plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
        plt.colorbar(); plt.pause(0.1)

        # 2) Filter ±2 m/s, ≤18 s
        kv, kt = (np.abs(vel) <= 2), (t <= 18)
        t2, v2, S2 = t[kt], vel[kv], S[kv, :][:, kt]

        f_kv = f[kv]

        # Save the S2 vector to a file
        os.makedirs("Spectrogram_S2", exist_ok=True)
        output_path_S2 = os.path.join("Spectrogram_S2", f"Spectrogram_vector_{file_base}.npy")
        np.save(output_path_S2, S2)
        print(f"Spectrogram (S2) vector saved to: {output_path_S2}")

        # Save the v2 vector to a file
        os.makedirs("Spectrogram_v2", exist_ok=True)
        output_path_v2 = os.path.join("Spectrogram_v2", f"Spectrogram_vector_{file_base}.npy")
        np.save(output_path_v2, v2)
        print(f"Spectrogram (v2) vector saved to: {output_path_v2}")
    
        # Save the t2 vector to a file
        os.makedirs("Spectrogram_t2", exist_ok=True)
        output_path_t2 = os.path.join("Spectrogram_t2", f"Spectrogram_vector_{file_base}.npy")
        np.save(output_path_t2, t2)
        print(f"Spectrogram (t2) vector saved to: {output_path_t2}")

        os.makedirs("Spectrogram_f_kv", exist_ok=True)
        output_path_f_kv = os.path.join("Spectrogram_f_kv", f"Spectrogram_vector_{file_base}.npy")
        np.save(output_path_f_kv, f_kv)
        print(f"Spectrogram (f[kv]) vector saved to: {output_path_f_kv}")


    # if .npy, find t2, v2, and S2 .npy files and put them into the appropriate vectors, then proceed
    elif fp.endswith('.npy'):
        print("Found .npy file, handling differently")
            # Define the paths to the .npy files containing the vectors
        v2_path = f"/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/Spectrogram_v2/{file_base}.npy"
        t2_path = f"/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/Spectrogram_t2/{file_base}.npy"
        S2_path = f"/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/Spectrogram_S2/{file_base}.npy"
        f_kv_path = f"/Users/karolinalow/Capstone/PicoScenes-Python-Toolbox/Spectrogram_f_kv/{file_base}.npy"

        try:
            v2 = np.load(v2_path)
            t2 = np.load(t2_path)
            S2 = np.load(S2_path)
            f_kv = np.load(f_kv_path)

            print(f"Loaded v2 from {v2_path}, shape: {v2.shape}")
            print(f"Loaded t2 from {t2_path}, shape: {t2.shape}")
            print(f"Loaded S2 from {S2_path}, shape: {S2.shape}")
            print(f"Loaded f[kv] from {f_kv_path}, shape: {f_kv.shape}")

        # Proceed with your logic here using v2, t2, S2

        except FileNotFoundError as e:
            print(f"Error loading one of the files: {e}")
    

    else:
        print("Unsupported file type")
    # end if

    plt.figure()
    plt.pcolormesh(t2, v2, S2, shading='auto')
    plt.title("Filtered Spectrogram (±2 m/s)")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
    plt.colorbar(); plt.pause(0.1)
   

    # 3) Torso speed
    p = extract_torso_speed(f_kv, S2)
    plt.figure()
    plt.plot(t2, p, linewidth=2)
    plt.title("Torso Speed")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
    plt.ylim([0,2]); plt.grid(True); plt.pause(0.1)

    # 4) Stable window
    st = find_stable_section(p)
    if st:
        # convert Python 0-based to MATLAB 1-based
        s1, e1 = st['start']+1, st['end']+1
        print(f"Section with least variance starts at index {s1} and ends at index {e1}")
        print(f"Average velocity of this section: {st['avg_velocity']:.4f}, Variance: {st['variance']:.6f}")

        # highlight
        plt.figure()
        plt.plot(t2, p, linewidth=2)
        idx = np.arange(st['start'], st['end']+1)
        plt.plot(t2[idx], p[idx], 'g', linewidth=4)
        plt.title("Torso Speed with Section of Least Variance Highlighted")
        plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
        plt.grid(True); plt.pause(0.1)

        # 5) Normalized Velocity Distribution
        nf = S2.shape[0]
        fd = np.zeros(nf)
        sj = int(np.ceil(nf*0.15))
        for j in range(sj, nf):
            fd[j] = np.mean(S2[j, :])
        df = f_kv[1] - f_kv[0]
        fdn = fd / (np.sum(fd) * df)
        os.makedirs("freq_dist", exist_ok=True)
        output_path1 = os.path.join("freq_dist", f"freq_dist_vector_{file_base}.npy")
        np.save(output_path1, fdn)
        print(f"Freq dist vector saved to: {output_path1}")


        plt.figure()
        plt.plot(v2, fdn, linewidth=2)
        plt.title("Normalized Velocity Distribution")
        plt.xlabel("Velocity (m/s)"); plt.ylabel("Probability Density")
        plt.grid(True); plt.pause(0.1)

        # 6) Autocorr
        ac, mac, sac, gac = compute_autocorr_features(S2, st['start'], st['end'])
        plt.figure()
        plt.hist(gac, bins=int(np.sqrt(len(gac))))
        plt.title("Histogram of Gradient of Autocorrelation Vector")
        plt.xlabel("Gradient Values"); plt.ylabel("Frequency")
        plt.grid(True); plt.pause(0.1)

        # 7) Torso-speed gradient
        seg = p[st['start']:st['end']+1]
        gts = np.gradient(seg)
        plt.figure()
        plt.hist(gts, bins=int(np.sqrt(len(gts))))
        plt.title("Histogram of Gradient of Torso Speed Vector")
        plt.xlabel("Gradient Values"); plt.ylabel("Frequency")
        plt.grid(True); plt.pause(0.1)

        # 8) Gait detection
        gc, pks = detect_gait(seg, t2[st['start']:st['end']+1])
        stride = np.mean(seg) * gc
        plt.figure()
        plt.plot(t2[st['start']:st['end']+1], seg, linewidth=2)
        if pks is not None:
            pts = t2[st['start']:st['end']+1][pks]
            plt.plot(pts, seg[pks], 'ro')
        plt.title("Torso Speed with Gait Cycle Peaks")
        plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
        plt.grid(True); plt.pause(0.1)

        # Print features in MATLAB order
        print(f"Auto Correlation: {ac:.4f}")
        print(f"Average Torso Speed (m/s): {np.mean(seg):.4f}")
        print(f"Average Gait Cycle Length (s): {gc:.4f}")
        print(f"Stride Length (m): {stride:.4f}")
        print(f"Histogram of Torso Speed Gradient Mean: {np.mean(gts):.4e}")
        print(f"Histogram of Torso Speed Gradient Std: {np.std(gts):.4f}")
        print(f"Histogram of AC Gradient Mean: {mac:.4f}")
        print(f"Histogram of AC Gradient Std: {sac:.4f}")

        # --- feature vector like MATLAB ---
    
        feature_vector = np.array([
            ac,
            np.mean(seg),
            gc,
            stride,
            np.mean(gts),
            np.std(gts),
            mac,
            sac
        ])
        print("feature_vector = [")
        for v in feature_vector:
                print(f"  {v:.4f}")
        print("]") 
        # Save the feature vector to a file
        os.makedirs("features", exist_ok=True)
        output_path = os.path.join("features", f"feature_vector_{file_base}.npy")
        np.save(output_path, feature_vector)
        print(f"Feature vector saved to: {output_path}")
    
    else:
        print("No valid sections found.")

    input("Close all figures, then press Enter to exit.\n")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()