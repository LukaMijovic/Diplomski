import matplotlib.pyplot as plt
import numpy as np

# Original data (without Neural Network)
algorithms = ["Linear Regression", "SVM"]
mae_original = [2506.310, 2470.957]
rmse_original = [5554.342, 5526.934]
r2_original = [0.078194, 0.087269]

# New data "after Fisher score"
mae_new_fisher = [2502.359, 2472.242]
rmse_new_fisher = [5532.996, 5527.320]
r2_new_fisher = [0.085266, 0.087142]

# New data "after MI"
mae_new_mi = [2507.207, 2486.840]
rmse_new_mi = [5541.343,  5540.862]
r2_new_mi = [0.082504, 0.082663]

# New data "after RFE"
mae_new_rfe = [2553.059, 2475.046]
rmse_new_rfe = [5606.002, 5531.536]
r2_new_rfe = [0.060968, 0.085749]

# Create figures and axes for each metric
fig1, ax1 = plt.subplots(figsize=(8, 6))
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Bar width
width = 0.2

# Plot 1: RMSE
x = np.arange(len(algorithms))
ax1.bar(x - 1.5 * width, rmse_original, width, label='Original', alpha=0.7, color='b')
ax1.bar(x - 0.5 * width, rmse_new_fisher, width, label='After Fisher Score', alpha=0.7, color='g')
ax1.bar(x + 0.5 * width, rmse_new_mi, width, label='After MI', alpha=0.7, color='r')
ax1.bar(x + 1.5 * width, rmse_new_rfe, width, label='After RFE', alpha=0.7, color='c')
ax1.set_ylabel('RMSE Values')
ax1.set_title('RMSE Comparison by Algorithm')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
ax1.set_ylim(5200, 5700)
ax1.legend()

# Plot 2: MAE
ax2.bar(x - 1.5 * width, mae_original, width, label='Original', alpha=0.7, color='b')
ax2.bar(x - 0.5 * width, mae_new_fisher, width, label='After Fisher Score', alpha=0.7, color='g')
ax2.bar(x + 0.5 * width, mae_new_mi, width, label='After MI', alpha=0.7, color='r')
ax2.bar(x + 1.5 * width, mae_new_rfe, width, label='After RFE', alpha=0.7, color='c')
ax2.set_ylabel('MAE Values')
ax2.set_title('MAE Comparison by Algorithm')
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms)
ax2.set_ylim(2000, 2600)
ax2.legend()

# Plot 3: R2
ax3.bar(x - 1.5 * width, r2_original, width, label='Original', alpha=0.7, color='b')
ax3.bar(x - 0.5 * width, r2_new_fisher, width, label='After Fisher Score', alpha=0.7, color='g')
ax3.bar(x + 0.5 * width, r2_new_mi, width, label='After MI', alpha=0.7, color='r')
ax3.bar(x + 1.5 * width, r2_new_rfe, width, label='After RFE', alpha=0.7, color='c')
ax3.set_ylabel('R2 Values')
ax3.set_title('R2 Comparison by Algorithm')
ax3.set_xticks(x)
ax3.set_xticklabels(algorithms)
ax3.set_ylim(0, 0.15)
ax3.legend()

# Show the plots
plt.tight_layout()
plt.show()
