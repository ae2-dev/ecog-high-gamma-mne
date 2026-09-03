import argparse
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.colors
from mne.stats import permutation_cluster_test, ttest_ind_no_p

from config import DEFAULT_OUTPUT_DIR, STAT_LATENCY, ALPHA, CLUSTER_ALPHA, SEED

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--n-permutations", type=int, default=500)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(args.work_dir / "tfr.npz", allow_pickle=True)
tfr_obj, tfr_face = d["tfr_obj"], d["tfr_face"]
tfr_times = d["tfr_times"]
freqs = d["freqs"]
ch_names = [str(ch) for ch in d["ch_names"]]

tt_sel = (tfr_times >= STAT_LATENCY[0]) & (tfr_times <= STAT_LATENCY[1])
n_obj, n_face = tfr_obj.shape[0], tfr_face.shape[0]
df = n_obj + n_face - 2
t_thresh = stats.t.ppf(1 - CLUSTER_ALPHA / 2, df)

t_obs_all, mask_all, sig = {}, {}, []
for ci, ch in enumerate(ch_names):
    X = [tfr_obj[:, ci][..., tt_sel], tfr_face[:, ci][..., tt_sel]]
    t_obs, clusters, p_vals, _ = permutation_cluster_test(
        X, threshold=t_thresh, n_permutations=args.n_permutations, tail=0,
        stat_fun=ttest_ind_no_p, adjacency=None, seed=SEED,
        out_type="indices", verbose=False)
    mask = np.zeros(t_obs.shape, bool)
    for cl, p in zip(clusters, p_vals):
        if p < ALPHA:
            mask[cl] = True
    t_obs_all[ch] = t_obs
    mask_all[ch] = mask
    if mask.any():
        sig.append(ch)
print(f"TFR: {len(sig)} significant channel(s): {sig}")

with open(args.work_dir / "tfr_stats_summary.txt", "w") as f:
    f.write(f"TFR significant channels: {sig}\n")

if sig:
    tsel = (tfr_times >= STAT_LATENCY[0]) & (tfr_times <= STAT_LATENCY[1])
    ncol = min(3, len(sig))
    nrow = int(np.ceil(len(sig) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4 * nrow), squeeze=False)
    fade_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fade", [(1, 1, 1, 0), (1, 1, 1, 1)])
    for ax, ch in zip(axes.ravel(), sig):
        t_full = np.zeros((len(freqs), len(tfr_times)))
        m_full = np.zeros_like(t_full, bool)
        t_full[:, tt_sel] = t_obs_all[ch]
        m_full[:, tt_sel] = mask_all[ch]
        zz = t_full[:, tsel]
        im = ax.pcolormesh(tfr_times[tsel], freqs, zz, vmin=-4, vmax=4, cmap="RdBu_r", shading="nearest")
        fade = np.where(m_full[:, tsel], 0.0, 0.6)
        ax.pcolormesh(tfr_times[tsel], freqs, fade, vmin=0, vmax=1, cmap=fade_cmap, shading="nearest")
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel="time (s)", ylabel="frequency (Hz)", title=f"t (object - face) - {ch}")
        plt.colorbar(im, ax=ax, label="t-value")
    for ax in axes.ravel()[len(sig):]:
        ax.set_axis_off()
    fig.tight_layout()
    out_path = args.work_dir / "08_TFR_stats_significant_channels.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")
else:
    print("no significant channels, skipping plot")

if args.show:
    plt.show()
