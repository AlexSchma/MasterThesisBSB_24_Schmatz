import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append(".")
from Code.DataParsing.DataUtils import family_wise_train_test_splitting
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import os
from Code.Models.CNNs._resnet import get_cnn
from Code.Models.CNNs.funprose_adapted_multi_class_len1000_13_17_kernel import get_funprose

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
default_dtype = torch.float32


# Load data
df = pd.read_csv('Data/Processed/simulated3_90000_1000_12_bp_motif.csv')

# One-hot encode sequences
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'T': [0, 0, 1, 0], 'G': [0, 0, 0, 1]} #jordi mapping

    return np.array([mapping[nuc] for nuc in seq], dtype=np.int8)

df['Sequence'] = df['Sequence'].apply(one_hot_encode)

# Map labels to integers
label_mapping = {'underexpressed': 0, 'no_change': 1, 'overexpressed': 2}
df['Label'] = df['Label'].map(label_mapping)

# Encoding treatments
TREATMENT_ENCODING = {
    "ABA": [1, 0, 0],
    "SA": [0, 1, 0],
    "JA": [0, 0, 1],
    "ABA_JA": [1, 0, 1],
    "SA_JA": [0, 1, 1],
    }

df['Treatment'] = df['Treatment'].apply(lambda x: TREATMENT_ENCODING[x])
df['Treatment'] = df['Treatment'].apply(np.array)

# Convert DataFrame columns to PyTorch tensors
sequences = [torch.tensor(s) for s in df['Sequence']]
labels = torch.tensor(df['Label'].values)
treatments = [torch.tensor(t) for t in df['Treatment']]

# Split data
X_train, X_temp, z_train, z_temp, y_train, y_temp = train_test_split(sequences, treatments, labels, test_size=0.4, random_state=42)
X_val, X_test, z_val, z_test, y_val, y_test = train_test_split(X_temp, z_temp, y_temp, test_size=0.5, random_state=42)

# Convert lists of tensors to tensor
X_train = torch.stack(X_train).float()
X_val = torch.stack(X_val).float()
X_test = torch.stack(X_test).float()
z_train = torch.stack(z_train).float()
z_val = torch.stack(z_val).float()
z_test = torch.stack(z_test).float()

# Create Tensor datasets
train_dataset = TensorDataset(X_train, z_train, y_train)
val_dataset = TensorDataset(X_val, z_val, y_val)
test_dataset = TensorDataset(X_test, z_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)








# model = get_cnn(False, device)
# model = get_cnn(False, device)
model = get_funprose().to(device)
# load model from checkpoint
#model.load_state_dict(torch.load("Code/Models/CNNs/FunProseAdapted_original_shape_model.pth", map_location=torch.device('cpu')))
def train(
    model,
   train_loader,
   val_loader,
    test_loader,
    device,
    path,
    name,
    num_epochs,
    batch_size,
    lr,
    weight_decay,
    only_index=False,
    data_npy=False,
):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters:", total_params)


    

    # if not os.path.exists(f"{path}/{name}_training_log.pkl"):
    #    epoch_logging = {}
    # else:
    #    with open(f"{path}/{name}_training_log.pkl", "rb") as f:
    #        epoch_logging = pickle.load(f)
    epoch_logging = {}

    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float("inf")

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    min_loss = np.inf
    early_stopper = EarlyStopper(patience=20, min_delta=0.01)
    for epoch in range(num_epochs):
        plt.close("all")
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        accuracies = []
        losses = []
        MCCs = []
        aucROC = []
        for batch in pbar:
            if only_index:
                batch_X = batch[0].transpose(1, 2).to(device)
            else:
                batch_X = batch[0].to(device)
                 # Batch of X sequence (batch, channel, sequence)
            batch_Z = batch[1].to(
                device
            )  # Batch of X condition
            batch_Y = batch[2].to(device, dtype=torch.long)
           
            optimizer.zero_grad()
            outputs = model(batch_X, batch_Z)
            # Assume outputs are logits from your model and batch_Y are the true class labels
            num_classes = outputs.size(1)  # outputs are [batch_size, num_classes]

            # Convert true class indices to one-hot encoding if necessary
            true_one_hot = torch.nn.functional.one_hot(batch_Y, num_classes=num_classes)

            # Calculate loss
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            # Convert outputs to probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Predicted class
            _, predicted = torch.max(outputs, 1)

            # Calculate metrics
            accuracies.append(accuracy_score(batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy()))
            losses.append(loss.item())
            MCCs.append(matthews_corrcoef(batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy()))

            # Compute ROC-AUC for each class and average
            roc_auc = {}
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(true_one_hot[:, i].detach().cpu().numpy(), probabilities[:, i].detach().cpu().numpy())
                roc_auc[i] = auc(fpr, tpr)
            average_auc_roc = np.mean(list(roc_auc.values()))

            aucROC.append(average_auc_roc)

            # Compute mean of metrics
            train_loss = np.mean(losses)
            train_accuracy = np.mean(accuracies)
            train_MCC = np.mean(MCCs)
            train_aucROC = np.mean(aucROC)

            pbar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Loss: {train_loss:.4f}, "
                f"Accuracy: {train_accuracy:.4f}, "
                f"MCC: {train_MCC:.4f}, "
                f"AUC-ROC: {train_aucROC:.4f}"
            )
            
        train_MCC = matthews_corrcoef(
            batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy()
        )
        # Validation passes
        with torch.no_grad():
            model.eval()
            pbar = tqdm(val_loader, total=len(val_loader))
            accuracies = []
            losses = []
            MCCs = []
            aucROC = []
            for batch in pbar:
                if only_index:
                    batch_X = batch[0].transpose(1, 2).to(device)
                else:
                    batch_X = batch[0].to(device)
                 # Batch of X sequence (batch, channel, sequence)
                batch_Z = batch[1].to(
                    device
                )  # Batch of X condition
                batch_Y = batch[2].to(device, dtype=torch.long)
                outputs = model(batch_X, batch_Z)
                loss = criterion(outputs, batch_Y)
                _, predicted = torch.max(outputs, 1)

                MCCs.append(
                    matthews_corrcoef(
                        batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy()
                    )
                )
                accuracies.append((predicted == batch_Y).sum().item() / batch_Y.size(0))
                losses.append(loss.item())

                # ROC-AUC adjustment for multi-class
                num_classes = outputs.size(1)
                true_one_hot = torch.nn.functional.one_hot(batch_Y, num_classes=num_classes)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
                auc_scores = [
                    roc_auc_score(true_one_hot[:, i].detach().cpu().numpy(), probabilities[:, i])
                    for i in range(num_classes)
                ]
                average_auc_roc = np.mean(auc_scores)
                aucROC.append(average_auc_roc)

                val_loss = np.mean(losses)
                val_accuracy = np.mean(accuracies)
                val_MCC = np.mean(MCCs)
                val_aucROC = np.mean(aucROC)


                pbar.set_description(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Val Loss: {np.mean(losses):.4f}, "
                    f"Val Accuracy: {np.mean(accuracies):.4f}, "
                    f"Val MCC: {np.mean(MCCs):.4f}, "
                    f"Val AUC-ROC: {np.mean(aucROC):.4f}"
                )

        epoch_logging[epoch] = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_MCC": train_MCC,
            "train_AUC-ROC": train_aucROC,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_MCC": val_MCC,
            "val_AUC-ROC": val_aucROC,
        }

        # save the dict
        with open(f"{path}/{name}_training_log.pkl", "wb") as f:
            pickle.dump(epoch_logging, f)
        # save the model
        torch.save(model.state_dict(), f"{path}/{name}_model.pth")

        # If validation loss is lower than min_loss, save the model
        if epoch > 0:
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(model.state_dict(), f"{path}/{name}_model_best_model.pth")
                model.eval()
    
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break
    
    # TEST SET
    with torch.no_grad():  # Check performance on test set of best model
        model.eval()
        all_predictions = []
        all_probas = []
        Y_test = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            if only_index:
                batch_X = batch[0].transpose(1, 2).to(device)
            else:
                batch_X = batch[0].to(device)
                 # Batch of X sequence (batch, channel, sequence)
            batch_Z = batch[1].to(
                device
            )  # Batch of X condition
            batch_Y = batch[2].to(device, dtype=torch.long)
            Y_test.append(batch_Y)

            outputs = model(batch_X, batch_Z)
            _, predicted = torch.max(outputs, 1)
            all_predictions.append(predicted)
            all_probas.append(outputs)
        print("Testing done!")
        all_predictions = torch.cat(all_predictions, dim=0)
        Y_test = torch.cat(Y_test, dim=0)
        all_probas = torch.cat(all_probas, dim=0)
        probabilities = torch.nn.functional.softmax(all_probas, dim=1).detach().cpu().numpy()
        
        accuracy = np.mean(all_predictions.detach().cpu().numpy() == Y_test)

        # Multi-class AUC-ROC
        auc_scores = roc_auc_score(Y_test, probabilities, multi_class="ovr")
        skplt.metrics.plot_roc(Y_test, probabilities, title="ROC Curve, \nclassification of differentially expressed genes", plot_macro=False, plot_micro=False)
        plt.savefig(f"{path}/{name}_roc_curve.png")

        # Precision-Recall Curve
        skplt.metrics.plot_precision_recall(Y_test, probabilities)
        plt.savefig(f"{path}/{name}_precision_recall_curve.png")
        
        # Calculate MCC
        mcc = matthews_corrcoef(Y_test, all_predictions.detach().cpu().numpy())
        print(f"Test MCC : {mcc:.4f}")
        print(f"Test AUC-ROC : {auc_scores:.4f}")
        print(f"Test Accuracy : {accuracy:.4f}")
        # Confusion Matrix
        skplt.metrics.plot_confusion_matrix(Y_test, all_predictions.detach().cpu().numpy(), normalize=False)
        plt.savefig(f"{path}/{name}_confusion_matrix.png")

        # Calibration plot
        # skplt.metrics.plot_calibration_curve(
        #     Y_test,
        #     [probabilities],  # List of probability predictions for each classifier (if comparing multiple models, extend this list)
        #     clf_names=[name]  # Names of the classifiers
        # )
        # plt.savefig(f"{path}/{name}_calibration_curve.png")

        if num_epochs == 0:
            return accuracy, mcc, auc_scores
    # plot loggings in 3 different plots, 1 row 3 columns
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 5))
    ax[0, 0].plot([epoch_logging[epoch]["train_loss"] for epoch in epoch_logging])
    ax[0, 0].set_title("Train Loss")
    ax[0, 1].plot([epoch_logging[epoch]["train_accuracy"] for epoch in epoch_logging])
    ax[0, 1].set_title("Train Accuracy")
    ax[0, 2].plot([epoch_logging[epoch]["train_MCC"] for epoch in epoch_logging])
    ax[0, 2].set_title("Train MCC")
    ax[0, 3].plot([epoch_logging[epoch]["train_AUC-ROC"] for epoch in epoch_logging])
    ax[0, 3].set_title("Train AUC-ROC")

    ax[1, 0].plot([epoch_logging[epoch]["val_loss"] for epoch in epoch_logging])
    ax[1, 0].set_title("Val Loss")
    ax[1, 1].plot([epoch_logging[epoch]["val_accuracy"] for epoch in epoch_logging])
    ax[1, 1].set_title("Val Accuracy")
    ax[1, 2].plot([epoch_logging[epoch]["val_MCC"] for epoch in epoch_logging])
    ax[1, 2].set_title("Val MCC")
    ax[1, 3].plot([epoch_logging[epoch]["val_AUC-ROC"] for epoch in epoch_logging])
    ax[1, 3].set_title("Val AUC-ROC")
    plt.tight_layout()
    plt.savefig(f"{path}/{name}_training.png")
    # Evaluation loop (on test set)


train(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    "Results/CNN/FunProse",
    "sim3_90000_1000_12_motif_13_17_kernel",
    300,
    512,
    0.00010229218879330196,
    0.0016447149582678627,
    only_index=True,
    #data_npy=np.load(
    #    "Data/Processed/Embeddings/embeddings_agro_nt_full_2024-03-18_13-03-59.npy"
    )
