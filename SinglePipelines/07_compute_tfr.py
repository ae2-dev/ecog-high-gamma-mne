import argparse
from pathlib import Path

import numpy as np
import matplotlib
import mne
from mne.time_frequency import tfr_array_morlet

from config import (DEFAULT_OUTPUT_DIR, CATEGORY, OBJECT_CODE, FACE_CODE,
                    CHANNEL_OF_INTEREST, TFR_BASELINE, PLOT_XLIM)

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--chunk-size", type=int, default=6)
parser.add_argument("--decim", type=int, default=10)
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

TFR_FREQS = np.concatenate([np.arange(4, 41, 2), np.arange(44, 101, 4), np.arange(108, 201, 8)]).astype(float)
tfr_times = times[::args.decim]
n_pad = len(times)
bl = (tfr_times >= TFR_BASELINE[0]) & (tfr_times <= TFR_BASELINE[1])

tfr = {}
for label, ep in [("object", epochs_obj), ("face", epochs_face)]:
    x = ep.get_data(copy=False)
    x = np.pad(x, ((0, 0), (0, 0), (n_pad, n_pad)))
    out = np.empty((x.shape[0], x.shape[1], len(TFR_FREQS), len(tfr_times)), np.float32)
    keep = slice(n_pad, n_pad + len(times), args.decim)
    for start in range(0, x.shape[1], args.chunk_size):
        sl = slice(start, start + args.chunk_size)
        power = tfr_array_morlet(x[:, sl], sfreq, TFR_FREQS, n_cycles=7,
                                 output="power", n_jobs=1)
        out[:, sl] = power[..., keep]
    base = out[..., bl].mean(axis=-1, keepdims=True)
    tfr[label] = (out - base) / base
tfr_obj, tfr_face = tfr["object"], tfr["face"]

np.savez(args.work_dir / "tfr.npz", tfr_obj=tfr_obj, tfr_face=tfr_face,
        tfr_times=tfr_times, freqs=TFR_FREQS, ch_names=ch_names)
print(f"saved {args.work_dir / 'tfr.npz'}")

tsel = (tfr_times >= PLOT_XLIM[0]) & (tfr_times <= PLOT_XLIM[1])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, label, x in [(axes[0], "objects", tfr_obj), (axes[1], "faces", tfr_face)]:
    zz = x[:, ch_idx].mean(0)[:, tsel]
    im = ax.pcolormesh(tfr_times[tsel], TFR_FREQS, zz, vmin=-10, vmax=10,
                       cmap="RdBu_r", shading="nearest")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="frequency (Hz)", title=f"TFR {label} - {CHANNEL_OF_INTEREST}")
    plt.colorbar(im, ax=ax, label="relative change")
fig.tight_layout()

out_path = args.work_dir / "07_TFR_IO03_object_vs_face.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"saved {out_path}")

if args.show:
    plt.show()
