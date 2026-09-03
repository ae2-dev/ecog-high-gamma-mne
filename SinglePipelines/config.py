from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "EEG Test Data" / "SubjectNY394" / "SubjectNY394"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

CATEGORY = {1: "false font", 2: "house", 3: "object", 4: "texture",
            5: "body", 6: "text", 7: "face"}
OBJECT_CODE = 3
FACE_CODE = 7

CHANNEL_OF_INTEREST = "IO03"
BAD_CHANNELS = ["G23"]
ERP_BASELINE = (-0.3, -0.05)
TFR_BASELINE = (-0.3, 0.05)
PLOT_XLIM = (-0.3, 0.6)
STAT_LATENCY = (0.0, 0.6)
ALPHA = 0.05
CLUSTER_ALPHA = 0.05
SEED = 42
