import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the results
with open(
    "Code/Models/CNNs/FunProseAdapted_original_shape_training_log.pkl", "rb"
) as f:
    epoch_logging = pickle.load(f)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))


ax[0].plot(
    [epoch_logging[epoch]["train_loss"] for epoch in epoch_logging],
    color="green",
    alpha=0.5,
    label="Train",
)
ax[0].plot(
    [epoch_logging[epoch]["val_loss"] for epoch in epoch_logging],
    color="blue",
    alpha=0.5,
    label="Validation",
)
ax[0].set_title("Training and Validation loss")
ax[1].plot(
    [epoch_logging[epoch]["train_accuracy"] for epoch in epoch_logging],
    color="green",
    alpha=0.5,
)
ax[1].plot(
    [epoch_logging[epoch]["val_accuracy"] for epoch in epoch_logging],
    color="blue",
    alpha=0.5,
)
ax[1].set_title("Training and Validation accuracy")
ax[2].plot(
    [epoch_logging[epoch]["train_MCC"] for epoch in epoch_logging],
    color="green",
    alpha=0.5,
)
ax[2].plot(
    [epoch_logging[epoch]["val_MCC"] for epoch in epoch_logging],
    color="blue",
    alpha=0.5,
)
ax[2].set_title("Training and Validation MCC")
ax[3].plot(
    [epoch_logging[epoch]["train_AUC-ROC"] for epoch in epoch_logging],
    color="green",
    alpha=0.5,
)
ax[3].plot(
    [epoch_logging[epoch]["val_AUC-ROC"] for epoch in epoch_logging],
    color="blue",
    alpha=0.5,
)
ax[3].set_title("Training and Validation AUC-ROC")

# Add legend to the first subplot
ax[0].legend()

plt.savefig("Code/Models/CNNs/FunProseAdapted_original_shape_training_log.png")
