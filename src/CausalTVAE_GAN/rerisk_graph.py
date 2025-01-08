import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Data
datasets = ['Car', 'Nursery']

# Updated Predicted and Ground Truth values for before, after, and difference
predicted_before = [8.56, 0.93]
ground_truth_before = [3.59, 0.93]

predicted_after = [3.70, 0.93]
ground_truth_after = [3.16, 0.93]

# Calculating differences
predicted_diff = [predicted_before[i] - predicted_after[i] for i in range(len(datasets))]
ground_truth_diff = [ground_truth_before[i] - ground_truth_after[i] for i in range(len(datasets))]

# Setting up the bar plot
bar_width = 0.35  # Width of the bars
x = np.arange(len(datasets))  # Label locations

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting for each dataset:
for i, dataset in enumerate(datasets):
    # Predicted bars (first group for each dataset)
    ax.bar(x[i] - bar_width / 2, predicted_before[i], bar_width, label='Proposed model (Before)' if i == 0 else "",
           color='blue', alpha=0.3)  # Transparent "Before" bar
    ax.bar(x[i] - bar_width / 2, predicted_after[i], bar_width, label='Proposed model (After)' if i == 0 else "",
           color='blue')  # Solid "After" bar

    # Ground Truth bars (second group for each dataset)
    ax.bar(x[i] + bar_width / 2, ground_truth_before[i], bar_width, label='Method 3 (Before)' if i == 0 else "",
           color='green', alpha=0.3)  # Transparent "Before" bar
    ax.bar(x[i] + bar_width / 2, ground_truth_after[i], bar_width, label='Method 3 (After)' if i == 0 else "",
           color='green')  # Solid "After" bar


    # Handle case where R0 equals Rf
    if predicted_before[i] == predicted_after[i]:
        ax.text(x[i] - bar_width / 2, predicted_after[i] + 0.3, f'$R_{{0}} = R_{{f}} = {predicted_after[i]:.2f}$',
                ha='center', color='black', fontsize=14)  # Display equality message
    else:
        # Annotating the bars with their respective values
        # Predicted bars
        ax.text(x[i] - bar_width / 2, predicted_before[i] + 0.2, f'$R_{{0}}={predicted_before[i]:.2f}$', ha='center',
                color='black', fontsize=14)  # Predicted Before (R0)
        ax.text(x[i] - bar_width / 2, predicted_after[i] + 0.3, f'$R_{{f}}={predicted_after[i]:.2f}$', ha='center',
                color='white', fontsize=14)  # Predicted After (Rf)
        # Placing the predicted diff value in the middle of the bar
        ax.text(x[i] - bar_width / 2, (predicted_before[i] + predicted_after[i]) / 2,
                f'$\Delta R={predicted_diff[i]:.2f}$', ha='center', color='black', fontsize=14)  # Predicted Diff value (Delta R)



    # Handle case where Ground Truth R0 equals Rf
    if ground_truth_before[i] == ground_truth_after[i]:
        ax.text(x[i] + bar_width / 2, ground_truth_after[i] + 0.3, f'$R_{{0}} = R_{{f}} = {ground_truth_after[i]:.2f}$',
                ha='center', color='black', fontsize=14)  # Display equality message
    else:
        # Ground Truth bars
        ax.text(x[i] + bar_width / 2, ground_truth_before[i] , f'$R_{{0}}={ground_truth_before[i]:.2f}$',
                ha='center',
                color='black', fontsize=14)  # Ground Truth Before (R0)
        ax.text(x[i] + bar_width / 2, ground_truth_after[i] - 0.3, f'$R_{{f}}={ground_truth_after[i]:.2f}$',
                ha='center',
                color='white', fontsize=14)  # Ground Truth After (Rf)
        # Placing the ground truth diff value in the middle of the bar
        ax.text(x[i] + bar_width / 2, (ground_truth_before[i] + ground_truth_after[i]) / 2 - 0.1,
                f'$\Delta R={ground_truth_diff[i]:.2f}$', ha='center', color='black', fontsize=14)  # Ground Truth Diff value (Delta R)

# Increase font size of axis labels and title
ax.set_xlabel('Dataset', fontsize=16)
ax.set_ylabel('Re-identification Risk (%)', fontsize=16)

# Increase font size for the x-tick labels (dataset names)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=14)  # Increase fontsize for dataset labels

# Optionally, if you want to adjust the font size of the y-tick labels (numeric values)
ax.tick_params(axis='y', labelsize=14)  # Adjust y-axis tick label font size
# Create a legend below the graph in a single row
handles, labels = ax.get_legend_handles_labels()
handles = handles[:3] + handles[3:5]  # Exclude the diff from the legend
ax.legend(handles, ['Proposed model (Before)', 'Proposed model (After)', 'Method 3 (Before)', 'Method 3 (After)'],
          loc='upper center', fontsize=12, bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=True)

# Adjust subplot parameters to set bottom and right margins
plt.subplots_adjust(bottom=0.3, right=0.5)

plt.tight_layout()
plt.show()
