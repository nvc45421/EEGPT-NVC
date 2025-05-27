import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Config ----
tag = "tiny3"  # Adjust to your desired tag
variant = "D"
version = 0  # Adjust to your desired version index

# ---- Locate metrics.csv ----
log_dir = Path(f"../pretrain/logs/EEGPT_{tag}_{variant}_csv")
metrics_path = log_dir / f"version_{version}" / "metrics.csv"

if not metrics_path.exists():
    raise FileNotFoundError(f"metrics.csv not found at: {metrics_path.resolve()}")

# ---- Load CSV ----
df = pd.read_csv(metrics_path)
print(f"Loaded metrics from: {metrics_path}")
print("Available columns:", df.columns.tolist())

# ---- Extract and Plot Loss Curves by Epoch ----
train_data = df[['epoch', 'train_loss']].dropna()
valid_data = df[['epoch', 'valid_loss']].dropna()

plt.figure(figsize=(10, 6))
plt.plot(train_data['epoch'], train_data['train_loss'], label='Train Loss', linewidth=2)
plt.plot(valid_data['epoch'], valid_data['valid_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curve by Epoch - EEGPT_{tag}_{variant}')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- Save Figure ----
output_path = log_dir / f"version_{version}" / f"loss_curve_epoch_{tag}_{variant}.png"
plt.savefig(output_path)
plt.close()

print(f"Saved loss curve to: {output_path}")
