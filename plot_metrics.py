import re
import matplotlib.pyplot as plt

def parse_logs(filename):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    with open(filename, 'r') as f:
        for line in f:
            if 'Train  loss=' in line:
                m = re.search(r'Train  loss=([\d\.]+)  acc=([\d\.]+)', line)
                if m:
                    train_losses.append(float(m.group(1)))
                    train_accs.append(float(m.group(2)))
            elif 'Val    loss=' in line:
                m = re.search(r'Val    loss=([\d\.]+)  acc=([\d\.]+)', line)
                if m:
                    val_losses.append(float(m.group(1)))
                    val_accs.append(float(m.group(2)))

    return train_losses, train_accs, val_losses, val_accs

train_losses, train_accs, val_losses, val_accs = parse_logs('training.log')
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
plt.plot(epochs, val_accs, 'r-o', label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
print("Successfully saved training_metrics.png")