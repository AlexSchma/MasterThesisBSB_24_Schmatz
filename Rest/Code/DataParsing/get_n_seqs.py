import pandas as pd
input_csv = 'Data/Processed/filtered_over_under_expressed.csv'
df = pd.read_csv(input_csv, index_col=0)
print(df.head())
columns = df.columns
Hormones = columns[2:]
print(Hormones)
count_dict = {}
for Hormone in Hormones:
    count_dict[Hormone] = df[Hormone].count()
    print(f"Number of rows with {Hormone}: {count_dict[Hormone]}")

Accs = {"SA": 0.7346, "ABA": 0.6179, "MeJA": 0.5287, "SA + MeJA": 0.6752, "ABA + MeJA": 0.6671}
AUCs = {"SA": 0.8118, "ABA": 0.6947, "MeJA": 0.5907, "SA + MeJA": 0.7631, "ABA + MeJA": 0.7195}

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Convert counts, Accs, and AUCs to lists
hormones_sorted = sorted(count_dict.keys(), key=lambda x: count_dict[x])
counts_sorted = np.array([count_dict[Hormone] for Hormone in hormones_sorted]).reshape(-1, 1)
accs_sorted = [Accs[Hormone] for Hormone in hormones_sorted]
aucs_sorted = [AUCs[Hormone] for Hormone in hormones_sorted]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the counts on the x-axis vs ACCs on the y-axis
ax1.set_xlabel('Counts')
ax1.set_ylabel('ACC', color='tab:green')
ax1.scatter(counts_sorted, accs_sorted, color='tab:green', label='ACC', marker='x')

# Perform linear regression for ACC
model_acc = LinearRegression().fit(counts_sorted, accs_sorted)
acc_line = model_acc.predict(counts_sorted)
ax1.plot(counts_sorted, acc_line, color='tab:green', linestyle='--')

ax1.tick_params(axis='y', labelcolor='tab:green')

# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('AUC', color='tab:red')
ax2.scatter(counts_sorted, aucs_sorted, color='tab:red', label='AUC', marker='o')

# Perform linear regression for AUC
model_auc = LinearRegression().fit(counts_sorted, aucs_sorted)
auc_line = model_auc.predict(counts_sorted)
ax2.plot(counts_sorted, auc_line, color='tab:red', linestyle='--')

ax2.tick_params(axis='y', labelcolor='tab:red')

# Add legends
fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add a legend for the different hormones
for i, hormone in enumerate(hormones_sorted):
    ax1.text(counts_sorted[i], accs_sorted[i], hormone, fontsize=9, ha='right')
    ax2.text(counts_sorted[i], aucs_sorted[i], hormone, fontsize=9, ha='left')

plt.title('Counts vs ACCs/AUCs for Different Hormones')
plt.show()



