import re
import matplotlib.pyplot as plt
import argparse
import os

# --- CLI args ---
parser = argparse.ArgumentParser(description="Plot training metrics from HipGPT log file")
parser.add_argument("log_file", nargs="?", default="out.log",
                    help="Path to training log file (default: out.log)")
parser.add_argument("--save", action="store_true",
                    help="Save plots as PNG instead of showing interactively")
parser.add_argument("--outdir", default="plots",
                    help="Directory to save plots if --save is used (default: plots/)")
args = parser.parse_args()

# --- Parse log file ---
pattern = re.compile(
    r"\[Step (\d+)\] Loss: ([\d.]+) \| Perplexity: ([\d.]+) \| Accuracy: ([\d.]+)%"
)

steps, losses, ppls, accs = [], [], [], []

if not os.path.exists(args.log_file):
    raise FileNotFoundError(f"Log file {args.log_file} not found!")

with open(args.log_file, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            ppls.append(float(m.group(3)))
            accs.append(float(m.group(4)))

# --- Plot helper ---
def plot_metric(x, y, label, ylabel, color, fname=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label, color=color)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(f"Training {label} over Time")
    plt.legend()
    plt.grid(True)
    if args.save:
        os.makedirs(args.outdir, exist_ok=True)
        outpath = os.path.join(args.outdir, fname)
        plt.savefig(outpath)
        print(f"Saved {label} plot to {outpath}")
    else:
        plt.show()

# --- Make plots ---
plot_metric(steps, losses, "Loss", "Loss", "blue", "loss.png")
plot_metric(steps, ppls, "Perplexity", "Perplexity", "orange", "perplexity.png")
plot_metric(steps, accs, "Accuracy (%)", "Accuracy (%)", "green", "accuracy.png")
