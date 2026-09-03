import argparse
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
from mne.stats import permutation_cluster_test, ttest_ind_no_p

from config import (DEFAULT_OUTPUT_DIR, PLOT_XLIM, STAT_LATENCY, ALPHA,
                    CLUSTER_ALPHA, SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--n-permutations", type=int, default=500)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(args.work_dir / "hgp.npz", allow_pickle=True)
hgp_obj, hgp_face = d["hgp_obj"], d["hgp_face"]
times = d["times"]
ch_names = [str(ch) for ch in d["ch_names"]]
n_obj, n_face = int(d["n_obj"]), int(d["n_face"])

t_sel = (times >= STAT_LATENCY[0]) & (times <= STAT_LATENCY[1])
df = n_obj + n_face - 2
t_thresh = stats.t.ppf(1 - CLUSTER_ALPHA / 2, df)

t_obs_all, mask_all, sig = {}, {}, []
for ci, ch in enumerate(ch_names):
    X = [hgp_obj[:, ci, t_sel], hgp_face[:, ci, t_sel]]
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
print(f"HGP: {len(sig)} significant channel(s): {sig}")

with open(args.work_dir / "hgp_stats_summary.txt", "w") as f:
    f.write(f"HGP significant channels: {sig}\n")

if sig:
    sel = (times >= PLOT_XLIM[0]) & (times <= PLOT_XLIM[1])
    ncol = min(3, len(sig))
    nrow = int(np.ceil(len(sig) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for ax, ch in zip(axes.ravel(), sig):
        ci = ch_names.index(ch)
        y_obj = hgp_obj[:, ci].mean(0)
        y_face = hgp_face[:, ci].mean(0)
        full_mask = np.zeros(len(times), bool)
        full_mask[t_sel] = mask_all[ch]
        ax.plot(times[sel], y_obj[sel], label=f"object (n={n_obj})", color="C0")
        ax.plot(times[sel], y_face[sel], label=f"face (n={n_face})", color="C1")
        lo, hi = ax.get_ylim()
        ax.fill_between(times[sel], lo, hi, where=full_mask[sel], color="gold", alpha=0.35,
                        label="p < 0.05 (cluster)")
        ax.set_ylim(lo, hi)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlim=PLOT_XLIM, xlabel="time (s)", ylabel="HGP (µV²·Hz²)", title=f"HGP - {ch}")
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes.ravel()[len(sig):]:
        ax.set_axis_off()
    fig.tight_layout()
    out_path = args.work_dir / "06_HGP_significant_channels.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")
else:
    print("no significant channels, skipping plot")

if args.show:
    plt.show()
