import sys
import numpy as np

sys.path.append('/Users/feiwang/Documents/VSCode/PicoScenes-Python-Toolbox') # Change this to your file path
from picoscenes import Picoscenes

def parse_csi_from_raw_csi_file(filepath):
    """
    Parses a .csi file where each frame in reader.raw contains a 4050-point CSI row.
    Filters invalid or malformed entries.

    Returns:
        np.ndarray: CSI matrix of shape (N, 4050)
    """
    reader = Picoscenes(filepath)
    csi_rows = []

    for frame in reader.raw:
        try:
            # Extract and force 1D array
            row = np.asarray(frame['CSI']['CSI']).flatten()
            if row.shape[0] == 4050:
                csi_rows.append(row)
        except Exception:
            continue  # Skip invalid or incomplete frames

    if len(csi_rows) == 0:
        raise RuntimeError("No valid CSI rows found.")

    return np.vstack(csi_rows).astype(np.complex64)
