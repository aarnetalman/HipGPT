import re
import matplotlib.pyplot as plt

# Path to your training log
log_file = "out.log"

# Regex to extract step, loss, ppl, acc
pattern = re.compile(
    r"\[Step (\d+)\] Loss: ([\d.]+) \| Perplexity: ([\d.]+) \| Accuracy: ([\d.]+)%"
)

steps, losses, ppls, accs = [], [], [], []

with open(log_file, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            ppl = float(m.group(3))
            acc = float(m.group(4))
            steps.append(step)
            losses.append(loss)
            ppls.append(ppl)
            accs.append(acc)

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot perplexity
plt.figure(figsize=(10, 6))
plt.plot(steps, ppls, label="Perplexity", color="orange")
plt.xlabel("Step")
plt.ylabel("Perplexity")
plt.title("Training Perplexity over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(steps, accs, label="Accuracy (%)", color="green")
plt.xlabel("Step")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy over Time")
plt.legend()
plt.grid(True)
plt.show()
