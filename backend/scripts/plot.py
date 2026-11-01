import matplotlib.pyplot as plt

# Data from your successful training run
epochs = list(range(1, 11))
losses = [0.2883, 0.0915, 0.0552, 0.0436, 0.0340, 0.0305, 0.0294, 0.0238, 0.0211, 0.00012]
accuracies = [0.8707, 0.9653, 0.9790, 0.9830, 0.9867, 0.9884, 0.9887, 0.9900, 0.9915, 0.9920]

# Create Figure
plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracies, marker='o', color='teal', linewidth=2)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--')

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, losses, marker='o', color='crimson', linewidth=2)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig('training_performance.png', dpi=300) # Save as high-res for report
plt.show()