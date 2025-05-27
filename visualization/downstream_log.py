import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ---- Config ----
tag = "large"        # e.g., "tiny1", "large"
variant = "D"        # e.g., "D"
task = "BCIC2B"      # e.g., "BCIC2A", "BCIC2B"

# ---- Main Log Directories ----
log_dir = Path(f"../downstream/logs/{tag}_{variant}_rep_EEGPT_{task}_csv")
original_root_dir = Path(f"../downstream/logs/EEGPT_{task}_csv_original")

# ---- Helper to Collect Accuracy from standard version_X structure ----
def collect_accuracies(directory, label=""):
    accs = []
    version_dirs = sorted([d for d in directory.iterdir() if d.is_dir() and re.match(r'version_\d+', d.name)])

    for version_dir in version_dirs:
        metrics_path = version_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"[Warning] Missing: {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        if 'valid_balanced_accuracy' in df.columns:
            last_acc = df['valid_balanced_accuracy'].dropna().iloc[-1]
            print(f"[{label}] {version_dir.name}: {last_acc:.4f}")
            accs.append(last_acc)
        else:
            print(f"[Warning] 'valid_balanced_accuracy' not in columns for: {metrics_path}")
    return accs

# ---- Helper to Collect Accuracy from subjectX/metrics.csv structure ----
def collect_subject_accuracies(root_dir, label="Original"):
    accs = []
    for subject_dir in sorted(root_dir.glob("subject*")):
        metrics_path = subject_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"[Warning] Missing: {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        if 'valid_balanced_accuracy' in df.columns:
            last_acc = df['valid_balanced_accuracy'].dropna().iloc[-1]
            print(f"[{label}] {subject_dir.name}: {last_acc:.4f}")
            accs.append(last_acc)
        else:
            print(f"[Warning] 'valid_balanced_accuracy' not in columns for: {metrics_path}")
    return accs

# ---- Collect from fine-tuned model ----
if not log_dir.exists():
    raise FileNotFoundError(f"Fine-tuned log directory not found: {log_dir.resolve()}")
ft_accuracies = collect_accuracies(log_dir, label="Fine-tune")

# ---- Collect from original (flat subjectX folders) ----
if original_root_dir.exists():
    orig_accuracies = collect_subject_accuracies(original_root_dir, label="Original")
else:
    print(f"[Warning] Original log directory not found: {original_root_dir.resolve()}")
    orig_accuracies = []

# ---- Summary ----
if ft_accuracies:
    avg_ft = sum(ft_accuracies) / len(ft_accuracies)
    print(f"\nAverage Fine-tuned valid_balanced_accuracy: {avg_ft:.4f}")

if orig_accuracies:
    avg_orig = sum(orig_accuracies) / len(orig_accuracies)
    print(f"Average Original valid_balanced_accuracy: {avg_orig:.4f}")

if not ft_accuracies and not orig_accuracies:
    print("No accuracy data found.")
