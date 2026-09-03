import argparse
from pathlib import Path

import numpy as np
import matplotlib
import mne
from mne.time_frequency import tfr_array_morlet

from config import (DEFAULT_OUTPUT_DIR, CATEGORY, OBJECT_CODE, FACE_CODE,
                    CHANNEL_OF_INTEREST, ERP_BASELINE, PLOT_XLIM)

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--chunk-size", type=int, default=6)
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

HG_FREQS = np.array([f for f in range(80, 195, 5) if f not in (120, 180)], float)
bl = (times >= ERP_BASELINE[0]) & (times <= ERP_BASELINE[1])

hgp = {}
for label, ep in [("object", epochs_obj), ("face", epochs_face)]:
    x = ep.get_data(copy=False)
    out = np.empty(x.shape, dtype=np.float32)
    for start in range(0, x.shape[1], args.chunk_size):
        sl = slice(start, start + args.chunk_size)
        power = tfr_array_morlet(x[:, sl], sfreq, HG_FREQS, n_cycles=10,
                                 output="power", n_jobs=1)
        power *= (HG_FREQS ** 2)[None, None, :, None]
        out[:, sl] = np.nanmean(power, axis=2)
    out -= out[:, :, bl].mean(axis=-1, keepdims=True)
    hgp[label] = out * 1e12
hgp_obj, hgp_face = hgp["object"], hgp["face"]

np.savez(args.work_dir / "hgp.npz", hgp_obj=hgp_obj, hgp_face=hgp_face,
        times=times, ch_names=ch_names, n_obj=n_obj, n_face=n_face)
print(f"saved {args.work_dir / 'hgp.npz'}")

sel = (times >= PLOT_XLIM[0]) & (times <= PLOT_XLIM[1])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times[sel], hgp_obj[:, ch_idx].mean(0)[sel], label=f"object (n={n_obj})", color="C0")
ax.plot(times[sel], hgp_face[:, ch_idx].mean(0)[sel], label=f"face (n={n_face})", color="C1")
ax.axvline(0, color="k", lw=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.set(xlim=PLOT_XLIM, xlabel="time (s)", ylabel="HGP (µV²·Hz², baseline-subtracted)",
      title=f"High-gamma power at {CHANNEL_OF_INTEREST}")
ax.legend(loc="upper left", fontsize=8)

out_path = args.work_dir / "04_HGP_IO03_object_vs_face.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"saved {out_path}")

if args.show:
    plt.show()
