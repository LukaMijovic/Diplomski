import matplotlib.pyplot as plt
import numpy as np

# Original data
algorithms = ["Linear Regression", "SVM", "Neural Network"]
mae_original = [2506.310, 2470.957, 2452.659]
rmse_original = [5554.342, 5526.934, 5563.248]
r2_original = [0.078194, 0.087269, 0.118026]

# New data "after fisher score"
mae_new = [2502.359, 2472.242, 2432.228]
rmse_new = [5532.996, 5527.320, 5432.715]
r2_new = [0.085266, 0.087142, 0.118123]

# Create figures and axes for each metric
fig1, ax1 = plt.subplots(figsize=(8, 6))
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Bar width
width = 0.35

# Plot 1: RMSE
x = np.arange(len(algorithms))
ax1.bar(x - width/2, rmse_original, width, label='RMSE Original', alpha=0.7, color='b')
ax1.bar(x + width/2, rmse_new, width, label='RMSE after Fisher Score', alpha=0.7, color='g')
ax1.set_ylabel('RMSE Values')
ax1.set_title('RMSE Comparison by Algorithm')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
ax1.set_ylim(5200, 5600)
ax1.legend()

# Plot 2: MAE
ax2.bar(x - width/2, mae_original, width, label='MAE Original', alpha=0.7, color='b')
ax2.bar(x + width/2, mae_new, width, label='MAE after Fisher Score', alpha=0.7, color='g')
ax2.set_ylabel('MAE Values')
ax2.set_title('MAE Comparison by Algorithm')
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms)
ax2.set_ylim(2000, 2600)
ax2.legend()

# Plot 3: R2
ax3.bar(x - width/2, r2_original, width, label='R2 Original', alpha=0.7, color='b')
ax3.bar(x + width/2, r2_new, width, label='R2 after Fisher Score', alpha=0.7, color='g')
ax3.set_ylabel('R2 Values')
ax3.set_title('R2 Comparison by Algorithm')
ax3.set_xticks(x)
ax3.set_xticklabels(algorithms)
ax3.set_ylim(0, 0.15)
ax3.legend()

# Show the plots
plt.tight_layout()
plt.show()
