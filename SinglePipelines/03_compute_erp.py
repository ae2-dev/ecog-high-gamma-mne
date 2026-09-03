import argparse
from pathlib import Path

import numpy as np
from scipy import signal
import matplotlib
import mne

from config import (DEFAULT_OUTPUT_DIR, CATEGORY, OBJECT_CODE, FACE_CODE,
                    CHANNEL_OF_INTEREST, ERP_BASELINE, PLOT_XLIM)

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")

epochs_clean = mne.read_epochs(args.work_dir / "epochs_clean-epo.fif", preload=True)
sfreq = epochs_clean.info["sfreq"]
times = epochs_clean.times
ch_names = epochs_clean.ch_names
ch_idx = ch_names.index(CHANNEL_OF_INTEREST)

epochs_obj = epochs_clean[CATEGORY[OBJECT_CODE]]
epochs_face = epochs_clean[CATEGORY[FACE_CODE]]
n_obj, n_face = len(epochs_obj), len(epochs_face)
bl = (times >= ERP_BASELINE[0]) & (times <= ERP_BASELINE[1])

erp = {}
for label, ep in [("object", epochs_obj), ("face", epochs_face)]:
    x = ep.get_data(copy=True)
    x = signal.detrend(x, axis=-1)
    x = mne.filter.filter_data(x, sfreq, 1.0, 30.0, method="iir",
                               iir_params=dict(order=4, ftype="butter"),
                               phase="zero", verbose=False)
    x -= x[:, :, bl].mean(axis=-1, keepdims=True)
    erp[label] = x
erp_obj, erp_face = erp["object"], erp["face"]

np.savez(args.work_dir / "erp.npz", erp_obj=erp_obj, erp_face=erp_face,
        times=times, ch_names=ch_names, n_obj=n_obj, n_face=n_face)
print(f"saved {args.work_dir / 'erp.npz'}")

sel = (times >= PLOT_XLIM[0]) & (times <= PLOT_XLIM[1])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times[sel], erp_obj[:, ch_idx].mean(0)[sel] * 1e6, label=f"object (n={n_obj})", color="C0")
ax.plot(times[sel], erp_face[:, ch_idx].mean(0)[sel] * 1e6, label=f"face (n={n_face})", color="C1")
ax.axvline(0, color="k", lw=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.set(xlim=PLOT_XLIM, xlabel="time (s)", ylabel="amplitude (µV)",
      title=f"ERP at {CHANNEL_OF_INTEREST}")
ax.legend(loc="upper left", fontsize=8)

out_path = args.work_dir / "03_ERP_IO03_object_vs_face.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"saved {out_path}")

if args.show:
    plt.show()
