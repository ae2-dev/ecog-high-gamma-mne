"""
ecog_ny_analysis.py
===================
MNE-Python reproduction of the FieldTrip example
"Analysis of high-gamma band signals in human ECoG"
https://www.fieldtriptoolbox.org/example/spectral/ecog_ny/

Every numbered section below mirrors a block of MATLAB code on that page.
All figures are written to  ecog_ny_analysis/figures/  as PNG files.

Run (from the eeg-analysis folder, or via the VS Code Run button):
    python ecog_ny_analysis/ecog_ny_analysis.py
Options:
    --show     also open the figures in interactive windows
    --fast     use 100 permutations instead of 500 (quick test run)
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal, stats
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")          # save PNGs only, no windows
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_morlet
from mne.stats import permutation_cluster_test, ttest_ind_no_p

mne.set_log_level("ERROR")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "EEG Test Data" / "SubjectNY394" / "SubjectNY394"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PERM = 100 if "--fast" in sys.argv else 500     # cfg.numrandomization
ALPHA = 0.05                                       # cfg.alpha
CLUSTER_ALPHA = 0.05                               # cfg.clusteralpha
SEED = 42

# The FieldTrip page lists the categories as
#   false font (3), house (4), object (5), texture (6), body (7), text (8), face (9)
# but the NY394_trl.mat that ships with the data uses codes 1..7 in the same
# order, so codes are shifted by 2.  Change these two numbers if your trl differs.
CATEGORY = {1: "false font", 2: "house", 3: "object", 4: "texture",
            5: "body", 6: "text", 7: "face"}
OBJECT_CODE = 3
FACE_CODE = 7

CHANNEL_OF_INTEREST = "IO03"      # 'EEG IO_03-REF' in the original file
BAD_CHANNELS = ["G23"]            # channel 23 marked bad in ft_rejectvisual
ERP_BASELINE = (-0.3, -0.05)      # cfg.baseline for ERP and HGP
TFR_BASELINE = (-0.3, 0.05)       # cfg.baseline for the TFR (as on the page)
PLOT_XLIM = (-0.3, 0.6)           # cfg.xlim
STAT_LATENCY = (0.0, 0.6)         # cfg.latency

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"          saved {path.relative_to(ROOT)}")
    if "--show" not in sys.argv:
        plt.close(fig)


def clean_name(name):
    """'EEG G_01-REF' -> 'G01'  (matches NY394_MRI_coor.txt)."""
    name = name.replace("-REF", "")
    for prefix in ("EEG ", "ECG ", "Pulse "):
        name = name.replace(prefix, "")
    return name.replace("_", "")


# ===========================================================================
# 1. Electrode positions on the pial surface
#    MATLAB: ft_read_headshape / ft_plot_mesh / ft_plot_sens
# ===========================================================================
log("1. Plotting electrodes on the pial surface")

coords, elec_type = {}, {}
for line in (DATA_DIR / "NY394_MRI_coor.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 5:
        coords[parts[0]] = np.array(parts[1:4], dtype=float)  # mm
        elec_type[parts[0]] = parts[4]

surf = sio.loadmat(DATA_DIR / "NY394_MRI_rh_pial_surface.mat")["surface"][0, 0]
pos = surf["pos"]                    # (n_vertices, 3) in mm
tri = surf["tri"].astype(int) - 1    # MATLAB 1-based -> 0-based

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=tri,
                color=(0.88, 0.85, 0.82), edgecolor="none", shade=True, linewidth=0,
                antialiased=False)
xyz = np.array(list(coords.values()))
# shift the markers a few mm laterally (+x) so they sit on top of the surface
ax.scatter(xyz[:, 0] + 5, xyz[:, 1], xyz[:, 2], c="k", s=18, depthshade=False)
for lab, (x, y, z) in coords.items():
    ax.text(x + 6, y, z, lab, fontsize=4.5, color="navy")
ax.view_init(elev=0, azim=0)         # lateral view of the right hemisphere
ax.set_box_aspect(np.ptp(pos, axis=0))
ax.set_axis_off()
ax.set_title("NY394 - electrodes on the right-hemisphere pial surface")
savefig(fig, "01_electrodes_on_brain.png")

# ===========================================================================
# 2. Load and segment the data
#    MATLAB: ft_preprocessing with cfg.trl from NY394_trl.mat
# ===========================================================================
log("2. Loading and segmenting the EDF")
raw = mne.io.read_raw_edf(DATA_DIR / "NY394_VisualLoc_R1.edf", preload=True)
sfreq = raw.info["sfreq"]
eeg_original = [ch for ch in raw.ch_names if ch.startswith("EEG ")]  # cfg.channel = 'EEG*'
raw.rename_channels(clean_name)
eeg_chs = [clean_name(ch) for ch in eeg_original]

ch_types = {}
for ch in raw.ch_names:
    if ch in elec_type:
        ch_types[ch] = "seeg" if elec_type[ch] == "D" else "ecog"
    elif ch in eeg_chs:
        ch_types[ch] = "ecog"          # intracranial contact without coordinates
    elif ch.startswith("EKG"):
        ch_types[ch] = "ecg"
    else:
        ch_types[ch] = "misc"
raw.set_channel_types(ch_types)
montage = mne.channels.make_dig_montage(
    ch_pos={ch: coords[ch] / 1000.0 for ch in raw.ch_names if ch in coords},
    coord_frame="mri")
raw.set_montage(montage, on_missing="ignore")

trl = sio.loadmat(DATA_DIR / "NY394_trl.mat")["trl"]
begsample, endsample, offset, cond = trl.T
onset = begsample - offset - 1                     # 1-based MATLAB -> 0-based
events = np.column_stack([onset, np.zeros_like(onset), cond]).astype(int)
tmin = offset[0] / sfreq                           # -0.5 s
tmax = (endsample[0] - begsample[0] + offset[0]) / sfreq   # +1.0 s
event_id = {CATEGORY[c]: int(c) for c in np.unique(cond)}

# ===========================================================================
# 3. Select the 'EEG*' channels
#    MATLAB: ft_selectdata with cfg.channel = 'EEG*'
# ===========================================================================
epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    baseline=None, picks=eeg_chs, preload=True)
del raw
log(f"   {len(epochs)} trials x {len(epochs.ch_names)} channels, "
    f"{tmin:.2f} to {tmax:.2f} s")

# ===========================================================================
# 4. Reject the bad channel
#    MATLAB: ft_rejectvisual, cfg.method = 'channel'  -> channel 23 removed
# ===========================================================================
log(f"4. Dropping bad channel(s): {BAD_CHANNELS}")
epochs.drop_channels(BAD_CHANNELS)

# ===========================================================================
# 5. Reject bad trials
#    MATLAB: ft_rejectvisual, cfg.method = 'summary' (interactive).
#    Here we do the same thing automatically: compute the per-trial
#    variance summary that the FieldTrip GUI shows, and drop trials that are
#    robust outliers (> 5 median absolute deviations above the median).
# ===========================================================================
log("5. Automatic trial rejection (variance summary)")
data = epochs.get_data(copy=False)
trial_var = data.var(axis=2).mean(axis=1)          # mean over channels of per-channel variance
med = np.median(trial_var)
mad = stats.median_abs_deviation(trial_var, scale="normal")
bad_trials = np.where(trial_var > med + 5 * mad)[0]
epochs_clean = epochs.copy().drop(bad_trials, reason="variance outlier")
log(f"   dropped {len(bad_trials)} of {len(epochs)} trials -> "
    f"{len(epochs_clean)} remain")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(trial_var * 1e12, "o", ms=3, label="kept")
ax.plot(bad_trials, trial_var[bad_trials] * 1e12, "rx", ms=7, label="rejected")
ax.axhline((med + 5 * mad) * 1e12, color="gray", ls="--", lw=1, label="threshold")
ax.set(xlabel="trial", ylabel="mean variance (µV²)", title="Trial summary (ft_rejectvisual equivalent)")
ax.legend()
savefig(fig, "02_trial_rejection_summary.png")

epochs_obj = epochs_clean[CATEGORY[OBJECT_CODE]]
epochs_face = epochs_clean[CATEGORY[FACE_CODE]]
n_obj, n_face = len(epochs_obj), len(epochs_face)
times = epochs_clean.times
ch_names = epochs_clean.ch_names
ch_idx = ch_names.index(CHANNEL_OF_INTEREST)
log(f"   {n_obj} object trials, {n_face} face trials")

# ===========================================================================
# 6-9. ERP: detrend, 1-30 Hz band-pass, baseline [-0.3 -0.05]
#    MATLAB: ft_timelockanalysis with cfg.preproc.{detrend,hpfilter,lpfilter}
#            + ft_timelockbaseline
# ===========================================================================
log("6. ERP: detrend + 1-30 Hz Butterworth + baseline")


def erp_preproc(ep):
    x = ep.get_data(copy=True)
    x = signal.detrend(x, axis=-1)                                   # cfg.preproc.detrend
    x = mne.filter.filter_data(x, sfreq, 1.0, 30.0, method="iir",   # hp 1 Hz, lp 30 Hz
                               iir_params=dict(order=4, ftype="butter"),
                               phase="zero", verbose=False)         # two-pass, like FieldTrip
    bl = (times >= ERP_BASELINE[0]) & (times <= ERP_BASELINE[1])
    x -= x[:, :, bl].mean(axis=-1, keepdims=True)                    # ft_timelockbaseline
    return x                                                         # (trials, ch, time)


erp_obj = erp_preproc(epochs_obj)
erp_face = erp_preproc(epochs_face)

# ===========================================================================
# 10. Plot the ERP at IO_03, objects vs faces
#    MATLAB: ft_singleplotER, cfg.channel = 'EEG IO_03-REF'
# ===========================================================================


def plot_er(ax, y_obj, y_face, title, ylabel, mask=None):
    """Objects vs faces time course, restricted to PLOT_XLIM (cfg.xlim) so the
    wavelet edge artefacts at the epoch borders do not dominate the y-axis."""
    sel = (times >= PLOT_XLIM[0]) & (times <= PLOT_XLIM[1])
    ax.plot(times[sel], y_obj[sel], label=f"object (n={n_obj})", color="C0")
    ax.plot(times[sel], y_face[sel], label=f"face (n={n_face})", color="C1")
    if mask is not None and mask.any():
        lo, hi = ax.get_ylim()
        ax.fill_between(times[sel], lo, hi, where=mask[sel], color="gold", alpha=0.35,
                        label="p < 0.05 (cluster)")
        ax.set_ylim(lo, hi)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlim=PLOT_XLIM, xlabel="time (s)", ylabel=ylabel, title=title)
    ax.legend(loc="upper left", fontsize=8)


fig, ax = plt.subplots(figsize=(7, 4))
plot_er(ax, erp_obj[:, ch_idx].mean(0) * 1e6, erp_face[:, ch_idx].mean(0) * 1e6,
        f"ERP at {CHANNEL_OF_INTEREST}", "amplitude (µV)")
savefig(fig, "03_ERP_IO03_object_vs_face.png")

# ===========================================================================
# 11-14. High-gamma power (HGP)
#    MATLAB: ft_freqanalysis cfg.method='tfr' (Morlet wavelets), width = 10,
#            foi = 80:5:190 without 120 and 180 (60 Hz harmonics),
#            then power * freq^2 (1/f correction), mean over freq,
#            baseline [-0.3 -0.05] subtraction.
# ===========================================================================
log("11. High-gamma power (Morlet 80-190 Hz, 10 cycles)")
HG_FREQS = np.array([f for f in range(80, 195, 5) if f not in (120, 180)], float)


def hgp_from_epochs(ep, chunk=6):
    """Per-trial high-gamma power, computed a few channels at a time to keep
    memory low.  Returns (n_trials, n_channels, n_times)."""
    x = ep.get_data(copy=False)
    out = np.empty(x.shape, dtype=np.float32)
    for start in range(0, x.shape[1], chunk):
        sl = slice(start, start + chunk)
        power = tfr_array_morlet(x[:, sl], sfreq, HG_FREQS, n_cycles=10,
                                 output="power", n_jobs=1)     # (trials, ch, freq, time)
        power *= (HG_FREQS ** 2)[None, None, :, None]           # freqcorr = freq.^2
        out[:, sl] = np.nanmean(power, axis=2)                  # nanmean over freq
    return out


def baseline_subtract(x, window):
    bl = (times >= window[0]) & (times <= window[1])
    return x - x[:, :, bl].mean(axis=-1, keepdims=True)


hgp_obj = baseline_subtract(hgp_from_epochs(epochs_obj), ERP_BASELINE) * 1e12   # V² -> µV²
hgp_face = baseline_subtract(hgp_from_epochs(epochs_face), ERP_BASELINE) * 1e12

# ===========================================================================
# 15. Plot HGP at IO_03
# ===========================================================================
fig, ax = plt.subplots(figsize=(7, 4))
plot_er(ax, hgp_obj[:, ch_idx].mean(0), hgp_face[:, ch_idx].mean(0),
        f"High-gamma power at {CHANNEL_OF_INTEREST}", "HGP (µV²·Hz², baseline-subtracted)")
savefig(fig, "04_HGP_IO03_object_vs_face.png")

# ===========================================================================
# 16-17. Cluster-based permutation statistics, objects vs faces
#    MATLAB: ft_timelockstatistics, cfg.method='montecarlo',
#            cfg.statistic='indepsamplesT', cfg.correctm='cluster',
#            cfg.neighbours=[] (no clustering across channels),
#            cfg.numrandomization=500, cfg.alpha=0.05, latency [0 0.6]
# ===========================================================================
log(f"16. Cluster permutation tests ({N_PERM} permutations per channel)")
t_sel = (times >= STAT_LATENCY[0]) & (times <= STAT_LATENCY[1])
stat_times = times[t_sel]
df = n_obj + n_face - 2
t_thresh = stats.t.ppf(1 - CLUSTER_ALPHA / 2, df)   # two-sided cluster-forming threshold


def cluster_test_per_channel(x_obj, x_face, label):
    """Run an independent-samples cluster test separately for every channel
    (cfg.neighbours = [] in FieldTrip).  Returns dict channel -> (t, mask, p_min)."""
    results = {}
    for ci, ch in enumerate(ch_names):
        X = [x_obj[:, ci], x_face[:, ci]]               # each (trials, [freq,] time)
        t_obs, clusters, p_vals, _ = permutation_cluster_test(
            X, threshold=t_thresh, n_permutations=N_PERM, tail=0,
            stat_fun=ttest_ind_no_p, adjacency=None, seed=SEED,
            out_type="indices", verbose=False)
        mask = np.zeros(t_obs.shape, bool)
        for cl, p in zip(clusters, p_vals):
            if p < ALPHA:
                mask[cl] = True
        results[ch] = (t_obs, mask, p_vals.min() if len(p_vals) else 1.0)
    sig = [ch for ch, (_, m, _) in results.items() if m.any()]
    log(f"   {label}: {len(sig)} significant channel(s): {sig}")
    return results, sig


stats_erp, sig_erp = cluster_test_per_channel(erp_obj[:, :, t_sel], erp_face[:, :, t_sel], "ERP")
stats_hgp, sig_hgp = cluster_test_per_channel(hgp_obj[:, :, t_sel], hgp_face[:, :, t_sel], "HGP")

# ===========================================================================
# 18-19. Plot HGP (and ERP, if any) for the significant channels with the
#        significant time points highlighted.   MATLAB: ft_singleplotER
#        with cfg.maskparameter = 'mask'
# ===========================================================================


def full_mask(short_mask):
    m = np.zeros(len(times), bool)
    m[t_sel] = short_mask
    return m


def plot_sig_channels(results, sig, x_obj, x_face, scale, ylabel, fname, title):
    if not sig:
        return
    n = len(sig)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for ax, ch in zip(axes.ravel(), sig):
        ci = ch_names.index(ch)
        plot_er(ax, x_obj[:, ci].mean(0) * scale, x_face[:, ci].mean(0) * scale,
                f"{title} - {ch}", ylabel, mask=full_mask(results[ch][1]))
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    savefig(fig, fname)


plot_sig_channels(stats_erp, sig_erp, erp_obj, erp_face, 1e6, "amplitude (µV)",
                  "05_ERP_significant_channels.png", "ERP")
plot_sig_channels(stats_hgp, sig_hgp, hgp_obj, hgp_face, 1.0, "HGP (µV²·Hz²)",
                  "06_HGP_significant_channels.png", "HGP")

# ===========================================================================
# 20-22. Broadband time-frequency representation, 4-200 Hz
#    MATLAB: ft_freqanalysis cfg.method='tfr', foi=[4:2:40 44:4:100 108:8:200],
#            toi=-0.3:0.02:0.6 (default width = 7 cycles),
#            ft_freqbaseline cfg.baselinetype='relchange', baseline [-0.3 0.05]
# ===========================================================================
log("20. Broadband TFR 4-200 Hz (Morlet, 7 cycles)")
TFR_FREQS = np.concatenate([np.arange(4, 41, 2), np.arange(44, 101, 4), np.arange(108, 201, 8)]).astype(float)
DECIM = 10                                    # 512 Hz / 10 ~= one sample every 0.0195 s (cfg.toi step 0.02)
tfr_times = times[::DECIM]


def tfr_from_epochs(ep, chunk=6):
    """Per-trial power 4-200 Hz.  A 7-cycle wavelet at 4 Hz is 1.75 s long,
    longer than the 1.5 s epoch, so (like FieldTrip does internally) we
    zero-pad each epoch on both sides before convolving and crop afterwards.
    Expect edge effects below ~10 Hz near the start/end of the epoch."""
    x = ep.get_data(copy=False)
    n_pad = len(times)                                   # pad one epoch length each side
    x = np.pad(x, ((0, 0), (0, 0), (n_pad, n_pad)))
    out = np.empty((x.shape[0], x.shape[1], len(TFR_FREQS), len(tfr_times)), np.float32)
    keep = slice(n_pad, n_pad + len(times), DECIM)       # samples of the original epoch
    for start in range(0, x.shape[1], chunk):
        sl = slice(start, start + chunk)
        power = tfr_array_morlet(x[:, sl], sfreq, TFR_FREQS, n_cycles=7,
                                 output="power", n_jobs=1)
        out[:, sl] = power[..., keep]
    return out


def relchange(x, window):
    bl = (tfr_times >= window[0]) & (tfr_times <= window[1])
    base = x[..., bl].mean(axis=-1, keepdims=True)
    return (x - base) / base


tfr_obj = relchange(tfr_from_epochs(epochs_obj), TFR_BASELINE)
tfr_face = relchange(tfr_from_epochs(epochs_face), TFR_BASELINE)

# ===========================================================================
# 23-24. Plot the TFR at IO_03 for objects and faces
#    MATLAB: ft_singleplotTFR, zlim [-10 10]
# ===========================================================================


def plot_tfr(ax, z, title, zlim, cmap="RdBu_r", cbar_label=None, mask=None, mask_alpha=0.4,
             xlim=PLOT_XLIM):
    tsel = (tfr_times >= xlim[0]) & (tfr_times <= xlim[1])
    zz = z[:, tsel]
    im = ax.pcolormesh(tfr_times[tsel], TFR_FREQS, zz, vmin=zlim[0], vmax=zlim[1],
                       cmap=cmap, shading="nearest")
    if mask is not None:
        # dim (fade towards white) everything that is NOT significant
        fade = np.where(mask[:, tsel], 0.0, 1 - mask_alpha)
        ax.pcolormesh(tfr_times[tsel], TFR_FREQS, fade, vmin=0, vmax=1,
                      cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                          "fade", [(1, 1, 1, 0), (1, 1, 1, 1)]),
                      shading="nearest")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="frequency (Hz)", title=title)
    plt.colorbar(im, ax=ax, label=cbar_label)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_tfr(axes[0], tfr_obj[:, ch_idx].mean(0), f"TFR objects - {CHANNEL_OF_INTEREST}",
         (-10, 10), cbar_label="relative change")
plot_tfr(axes[1], tfr_face[:, ch_idx].mean(0), f"TFR faces - {CHANNEL_OF_INTEREST}",
         (-10, 10), cbar_label="relative change")
fig.tight_layout()
savefig(fig, "07_TFR_IO03_object_vs_face.png")

# ===========================================================================
# 25. Cluster statistics on the TFR (clusters span time AND frequency,
#     but not channels)         MATLAB: ft_freqstatistics
# ===========================================================================
log(f"25. Cluster permutation tests on the TFR ({N_PERM} permutations per channel)")
tt_sel = (tfr_times >= STAT_LATENCY[0]) & (tfr_times <= STAT_LATENCY[1])
stats_tfr, sig_tfr = cluster_test_per_channel(tfr_obj[..., tt_sel], tfr_face[..., tt_sel], "TFR")

# ===========================================================================
# 26. Plot the t-statistic maps with the significant clusters opaque
#     MATLAB: ft_singleplotTFR, cfg.parameter='stat', maskalpha 0.4, zlim [-4 4]
# ===========================================================================
if sig_tfr:
    n = len(sig_tfr)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4 * nrow), squeeze=False)
    for ax, ch in zip(axes.ravel(), sig_tfr):
        t_obs, mask, _ = stats_tfr[ch]
        t_full = np.zeros((len(TFR_FREQS), len(tfr_times)))
        m_full = np.zeros_like(t_full, bool)
        t_full[:, tt_sel], m_full[:, tt_sel] = t_obs, mask
        plot_tfr(ax, t_full, f"t (object - face) - {ch}", (-4, 4), cbar_label="t-value",
                 mask=m_full, xlim=STAT_LATENCY)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    savefig(fig, "08_TFR_stats_significant_channels.png")

# ---------------------------------------------------------------------------
with open(FIG_DIR / "summary.txt", "w") as f:
    f.write(f"trials kept: {len(epochs_clean)} / {len(epochs)}  (rejected {list(bad_trials)})\n")
    f.write(f"object trials: {n_obj}, face trials: {n_face}\n")
    f.write(f"ERP significant channels: {sig_erp}\n")
    f.write(f"HGP significant channels: {sig_hgp}\n")
    f.write(f"TFR significant channels: {sig_tfr}\n")
log("Done.")
if "--show" in sys.argv:
    plt.show()
