import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib

from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

args.out_dir.mkdir(parents=True, exist_ok=True)

coords = {}
for line in (args.data_dir / "NY394_MRI_coor.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 5:
        coords[parts[0]] = np.array(parts[1:4], dtype=float)

surf = sio.loadmat(args.data_dir / "NY394_MRI_rh_pial_surface.mat")["surface"][0, 0]
pos = surf["pos"]
tri = surf["tri"].astype(int) - 1

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=tri,
                color=(0.88, 0.85, 0.82), edgecolor="none", shade=True, linewidth=0,
                antialiased=False)
xyz = np.array(list(coords.values()))
ax.scatter(xyz[:, 0] + 5, xyz[:, 1], xyz[:, 2], c="k", s=18, depthshade=False)
for lab, (x, y, z) in coords.items():
    ax.text(x + 6, y, z, lab, fontsize=4.5, color="navy")
ax.view_init(elev=0, azim=0)
ax.set_box_aspect(np.ptp(pos, axis=0))
ax.set_axis_off()
ax.set_title("NY394 - electrodes on the right-hemisphere pial surface")

out_path = args.out_dir / "01_electrodes_on_brain.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"saved {out_path}")

if args.show:
    plt.show()
