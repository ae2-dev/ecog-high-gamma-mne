import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib
import mne

from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, CATEGORY, BAD_CHANNELS

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
args.out_dir.mkdir(parents=True, exist_ok=True)

coords, elec_type = {}, {}
for line in (args.data_dir / "NY394_MRI_coor.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 5:
        coords[parts[0]] = np.array(parts[1:4], dtype=float)
        elec_type[parts[0]] = parts[4]

raw = mne.io.read_raw_edf(args.data_dir / "NY394_VisualLoc_R1.edf", preload=True)
sfreq = raw.info["sfreq"]
eeg_original = [ch for ch in raw.ch_names if ch.startswith("EEG ")]

rename_map = {}
for ch in raw.ch_names:
    new = ch.replace("-REF", "")
    for prefix in ("EEG ", "ECG ", "Pulse "):
        new = new.replace(prefix, "")
    rename_map[ch] = new.replace("_", "")
raw.rename_channels(rename_map)
eeg_chs = [rename_map[ch] for ch in eeg_original]

ch_types = {}
for ch in raw.ch_names:
    if ch in elec_type:
        ch_types[ch] = "seeg" if elec_type[ch] == "D" else "ecog"
    elif ch in eeg_chs:
        ch_types[ch] = "ecog"
    elif ch.startswith("EKG"):
        ch_types[ch] = "ecg"
    else:
        ch_types[ch] = "misc"
raw.set_channel_types(ch_types)

montage = mne.channels.make_dig_montage(
    ch_pos={ch: coords[ch] / 1000.0 for ch in raw.ch_names if ch in coords},
    coord_frame="mri")
raw.set_montage(montage, on_missing="ignore")

trl = sio.loadmat(args.data_dir / "NY394_trl.mat")["trl"]
begsample, endsample, offset, cond = trl.T
onset = begsample - offset - 1
events = np.column_stack([onset, np.zeros_like(onset), cond]).astype(int)
tmin = offset[0] / sfreq
tmax = (endsample[0] - begsample[0] + offset[0]) / sfreq
event_id = {CATEGORY[c]: int(c) for c in np.unique(cond)}

epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    baseline=None, picks=eeg_chs, preload=True)
del raw
print(f"{len(epochs)} trials x {len(epochs.ch_names)} channels, {tmin:.2f} to {tmax:.2f} s")

epochs.drop_channels(BAD_CHANNELS)
print(f"dropped bad channel(s): {BAD_CHANNELS}")

data = epochs.get_data(copy=False)
trial_var = data.var(axis=2).mean(axis=1)
med = np.median(trial_var)
mad = stats.median_abs_deviation(trial_var, scale="normal")
bad_trials = np.where(trial_var > med + 5 * mad)[0]
epochs_clean = epochs.copy().drop(bad_trials, reason="variance outlier")
print(f"dropped {len(bad_trials)} of {len(epochs)} trials -> {len(epochs_clean)} remain")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(trial_var * 1e12, "o", ms=3, label="kept")
ax.plot(bad_trials, trial_var[bad_trials] * 1e12, "rx", ms=7, label="rejected")
ax.axhline((med + 5 * mad) * 1e12, color="gray", ls="--", lw=1, label="threshold")
ax.set(xlabel="trial", ylabel="mean variance (µV²)", title="Trial summary (ft_rejectvisual equivalent)")
ax.legend()
fig.savefig(args.out_dir / "02_trial_rejection_summary.pdf", bbox_inches="tight")

epochs_clean.save(args.out_dir / "epochs_clean-epo.fif", overwrite=True)

with open(args.out_dir / "epochs_summary.txt", "w") as f:
    f.write(f"trials kept: {len(epochs_clean)} / {len(epochs)} (rejected {list(bad_trials)})\n")

print(f"saved {args.out_dir / 'epochs_clean-epo.fif'}")

if args.show:
    plt.show()
