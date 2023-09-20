import matplotlib.pyplot as plt
import numpy as np

# Data
algorithms = ["Linear Regression", "SVM", "Neural Network"]
mae = [2506.310, 2470.957, 2452.659]
rmse = [5554.342, 5526.934, 5563.248]
r2 = [0.078194, 0.087269, 0.118026]

# Colors for each algorithm
colors = ['b', 'g', 'r']

# Create figures and axes for each metric
fig1, ax1 = plt.subplots(figsize=(8, 6))
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Bar width
width = 0.35

# Plot 1: RMSE
x = np.arange(len(algorithms))
for i, color in enumerate(colors):
    ax1.bar(x[i], rmse[i], width, label=algorithms[i], alpha=0.7, color=color)
ax1.set_ylabel('RMSE Values')
ax1.set_title('RMSE by Algorithm')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
ax1.set_ylim(5200, 5600)
ax1.legend()

# Plot 2: MAE
for i, color in enumerate(colors):
    ax2.bar(x[i], mae[i], width, label=algorithms[i], alpha=0.7, color=color)
ax2.set_ylabel('MAE Values')
ax2.set_title('MAE by Algorithm')
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms)
ax2.set_ylim(2400, 2550)
ax2.legend()

# Plot 3: R2
for i, color in enumerate(colors):
    ax3.bar(x[i], r2[i], width, label=algorithms[i], alpha=0.7, color=color)
ax3.set_ylabel('R2 Values')
ax3.set_title('R2 by Algorithm')
ax3.set_xticks(x)
ax3.set_xticklabels(algorithms)
ax3.set_ylim(0.04, 0.12)
ax3.legend()

# Show the plots
plt.tight_layout()
plt.show()
