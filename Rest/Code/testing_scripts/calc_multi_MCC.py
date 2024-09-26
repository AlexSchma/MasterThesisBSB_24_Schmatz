import numpy as np
from sklearn.metrics import matthews_corrcoef

# Data from the table, representing the sum of each class
data = {
    "ABA": [26445, 3684, 3466],
    "ABA + MeJA": [23767, 5475, 4353],
    "MeJA": [30066, 1846, 1683],
    "SA": [24217, 5015, 4363],
    "SA + MeJA": [24538, 4717, 4340]
}

# Combine the data from all conditions into one list of true labels
# Each label is repeated based on the total count in each condition
y_true = (
    [0] * sum([data[cond][0] for cond in data]) +  # Class 1
    [1] * sum([data[cond][1] for cond in data]) +  # Class 2
    [2] * sum([data[cond][2] for cond in data])    # Class 3
)

def random_guess_mcc(y_true):
    """
    Calculate the MCC for random guesses based on class proportions.
    """
    # Count the occurrences of each class
    unique, counts = np.unique(y_true, return_counts=True)
    class_proportions = counts / len(y_true)
    
    # Create random predictions based on class proportions
    y_pred = np.random.choice(unique, size=len(y_true), p=class_proportions)
    
    # Calculate MCC for random guess
    mcc_random = matthews_corrcoef(y_true, y_pred)
    
    return mcc_random

def true_worst_case_mcc(y_true):
    """
    Calculate the true worst-case MCC by assigning incorrect predictions
    where no class label is predicted correctly.
    """
    y_pred = np.copy(y_true)
    
    # Assign wrong labels explicitly
    for i in range(len(y_true)):
        if y_true[i] == 0:
            y_pred[i] = 1  # Assign a different class (wrong prediction)
        elif y_true[i] == 1:
            y_pred[i] = 2  # Assign a different class (wrong prediction)
        elif y_true[i] == 2:
            y_pred[i] = 0  # Assign a different class (wrong prediction)
    
    # Calculate MCC for the worst-case
    mcc_min = matthews_corrcoef(y_true, y_pred)
    
    return mcc_min


# Calculate MCC for random guess and worst case
mcc_random = random_guess_mcc(y_true)
mcc_min = true_worst_case_mcc(y_true)

print(f"Random Guess MCC: {mcc_random}")
print(f"Worst Case MCC: {mcc_min}")
