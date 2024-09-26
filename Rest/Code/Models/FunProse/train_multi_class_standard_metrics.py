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
from sklearn.preprocessing import label_binarize


import os
from Code.Models.CNNs._resnet import get_cnn
from Code.Models.CNNs.funprose_adapted_multi_class import get_funprose

i = int(sys.argv[1])

# Use the parameter value in your script
print("The value of i is:", i)
def compute_metrics(outputs, batch_Y):
    num_classes = outputs.size(1)
    # Convert to probabilities with softmax
    probabilities = F.softmax(outputs, dim=1).detach().cpu().numpy()
    
    # Binarize the true labels for multi-class AUC-ROC
    true_one_hot = label_binarize(batch_Y.detach().cpu().numpy(), classes=np.arange(num_classes))

    # Calculate accuracy
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy())

    # Calculate MCC
    mcc = matthews_corrcoef(batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy())

    # Calculate multi-class ROC-AUC using One-vs-One strategy
    average_auc_roc = roc_auc_score(true_one_hot, probabilities, multi_class="ovo")

    return accuracy, mcc, average_auc_roc


torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
default_dtype = torch.float32

print(f"Using device: {device}")
# Load XY format one-hot encoded data
with open("Data/Processed/XY_formats/One_hot_encoded_up_down_multi_pytorch.pkl", "rb") as f:
    xy_dict = pickle.load(f)

print("Data loaded")
# load the gene names
gene_names = np.load("Data/Processed/XY_formats/gene_names_one_hot.npy")
print("Gene names loaded")
#test 12345
#test1 12346
#test2 12347

print(f"Starting iteration {i}")
random_seed = 12345 + i
np.random.seed(random_seed)
# Select randomly 30000 instances index
# random_index = np.random.choice(range(len(gene_names)), 3000, replace=False)

X_sequence = np.array(
    [xy_dict[gene][0][0].T for gene in xy_dict]
).squeeze()  # [random_index]
X_treatment = np.array([xy_dict[gene][0][1] for gene in xy_dict])  # [random_index]
Y = np.array([xy_dict[gene][1] for gene in xy_dict])  # [random_index]

# Get family names per gene
# load gene families
gene_families = pd.read_csv("Data/Processed/gene_families.csv", index_col=1)

# Map gene names to family names
# I cannot use a dictionary because the genes are repeated 5 times, eachper treatment
# I will use a list instead
family_names = []
for gene in gene_names:
    family_names.append(gene_families.loc[gene, "family_id"])


# Split the data
use_family_wise_split = True  # Set this to False for random splitting
print(f"Using family-wise splitting: {use_family_wise_split}")
if use_family_wise_split:
    # Family-wise splitting
    random_state = 1 + i
    train_index,test_index  = family_wise_train_test_splitting(
        np.array(family_names), random_state=random_state, test_size=0.1
    )
    
    # Get validation set from train set
    random_state = 2 + i
    train_index, val_index = family_wise_train_test_splitting(
        np.array(family_names)[train_index],
        random_state=random_state,
        test_size=0.23,
    )
else:
    # Random splitting (disregarding protein families)
    from sklearn.model_selection import train_test_split

    # Perform random splitting for train/test sets
    indices = np.arange(len(family_names))
    random_state = 1 + i
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=random_state
    )

    # Perform random splitting for validation set
    random_state = 2 + i
    train_index, val_index = train_test_split(
        train_index, test_size=0.3, random_state=random_state
    )

# X_sequence = torch.tensor(X_sequence, dtype=torch.default_dtype)
# X_treatment = torch.tensor(X_treatment, dtype=torch.default_dtype)
# Y = torch.tensor(Y, dtype=torch.long)
X_train_sequence = X_sequence[train_index]
X_test_sequence = X_sequence[test_index]
X_train_treatment = X_treatment[train_index]
X_test_treatment = X_treatment[test_index]
Y_train = Y[train_index]
Y_test = Y[test_index]
X_val_sequence = X_sequence[val_index]
X_val_treatment = X_treatment[val_index]
Y_val = Y[val_index]

print("Data set sizes:")
print(f"Train: {len(X_train_sequence)}")
print(f"Validation: {len(X_val_sequence)}")
print(f"Test: {len(X_test_sequence)}")

torch.set_float32_matmul_precision('high')

# model = get_cnn(False, device)
# model = get_cnn(False, device)
model = torch.compile(get_funprose().to(device))
# load model from checkpoint
#model.load_state_dict(torch.load("Code/Models/CNNs/FunProseAdapted_original_shape_model.pth", map_location=torch.device('cpu')))
def train(
    model,
    X_train_sequence,
    X_train_treatment,
    Y_train,
    X_val_sequence,
    X_val_treatment,
    Y_val,
    X_test_sequence,
    X_test_treatment,
    Y_test,
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

    class CustomDataset(Dataset):
        def __init__(self, X, Z, Y, transform=None):
            self.X = X
            self.Z = Z
            self.Y = Y
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            sample = {"X": self.X[idx], "Z": self.Z[idx], "Y": self.Y[idx]}

            if self.transform:
                sample = self.transform(sample)

            return sample

    def custom_collate(batch):
        batch_X = np.stack([sample["X"] for sample in batch])
        batch_Z = np.stack([sample["Z"] for sample in batch])
        batch_Y = np.stack([sample["Y"] for sample in batch])
        return {"X": batch_X, "Z": batch_Z, "Y": batch_Y}

    custom_dataset_train = CustomDataset(X_train_sequence, X_train_treatment, Y_train)

    data_loader_train = DataLoader(
        custom_dataset_train,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=True,
    )
    custom_dataset_val = CustomDataset(X_val_sequence, X_val_treatment, Y_val)
    data_loader_val = DataLoader(
        custom_dataset_val,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=False,
    )

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
        pbar = tqdm(data_loader_train, total=len(data_loader_train))
        accuracies = []
        losses = []
        MCCs = []
        aucROC = []
        all_outputs = []
        all_labels = []
        for batch in pbar:
            if only_index:
                batch_X = (
                    torch.tensor(data_npy[batch["X"]], dtype=default_dtype)
                    .to(device)
                    .transpose(1, 2)
                )  # Batch of X sequence (batch, channel, sequence)
            else:
                batch_X = torch.tensor(batch["X"], dtype=default_dtype).to(
                    device
                )  # Batch of X sequence
            batch_Z = torch.tensor(batch["Z"], dtype=default_dtype).to(
                device
            )  # Batch of X condition
            batch_Y = torch.tensor(batch["Y"], dtype=torch.long).to(
                device
            )  # Batch of label
            all_labels.append(batch_Y)

            optimizer.zero_grad()
            outputs = model(batch_X, batch_Z)
            all_outputs.append(outputs)
            accuracy, mcc, avg_auc_roc = compute_metrics(outputs, batch_Y)
            accuracies.append(accuracy)
            MCCs.append(mcc)
            aucROC.append(avg_auc_roc)

            # Calculate loss
            loss = criterion(outputs, batch_Y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

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
        epoch_train_accuracy, epoch_train_mcc, epoch_train_aucROC = compute_metrics(
            torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0))
        print(f"Overall Metrics Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {epoch_train_accuracy:.4f}, Train MCC: {epoch_train_mcc:.4f}, Train AUC-ROC: {epoch_train_aucROC:.4f}")

        # Validation passes
        with torch.no_grad():
            model.eval()
            pbar = tqdm(data_loader_val, total=len(data_loader_val))
            accuracies = []
            losses = []
            MCCs = []
            aucROC = []
            all_outputs = []
            all_labels = []
            for batch in pbar:
                if only_index:
                    batch_X = (
                        torch.tensor(data_npy[batch["X"]], dtype=default_dtype)
                        .to(device)
                        .transpose(1, 2)
                    )  # Batch of X sequence (batch, channel, sequence)
                else:
                    batch_X = torch.tensor(batch["X"], dtype=default_dtype).to(
                        device
                    )  # Batch of X sequence
                batch_Z = torch.tensor(batch["Z"], dtype=default_dtype).to(device)
                batch_Y = torch.tensor(batch["Y"], dtype=torch.long).to(device)
                all_labels.append(batch_Y)
                outputs = model(batch_X, batch_Z)
                all_outputs.append(outputs)
                loss = criterion(outputs, batch_Y)
                _, predicted = torch.max(outputs, 1)

                
                losses.append(loss.item())

                # ROC-AUC adjustment for multi-class
                accuracy, mcc, avg_auc_roc = compute_metrics(outputs, batch_Y)
                accuracies.append(accuracy)
                MCCs.append(mcc)
                aucROC.append(avg_auc_roc)

                aucROC.append(avg_auc_roc)

                val_loss = np.mean(losses)
                val_accuracy = np.mean(accuracies)
                val_MCC = np.mean(MCCs)
                val_aucROC = np.mean(aucROC)


                pbar.set_description(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}, "
                    f"Val MCC: {val_MCC:.4f}, "
                    f"Val AUC-ROC: {val_aucROC :.4f}"
                )
            epoch_val_accuracy, epoch_val_mcc, epoch_val_aucROC = compute_metrics(
                torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0))
            print(f"Overall Metrics Epoch {epoch + 1}/{num_epochs}, Val Accuracy: {epoch_val_accuracy:.4f}, Val MCC: {epoch_val_mcc:.4f}, Val AUC-ROC: {epoch_val_aucROC:.4f}")

        epoch_logging[epoch] = {
            "train_loss": train_loss,
            "train_accuracy": epoch_train_accuracy,
            "train_MCC": epoch_train_mcc,
            "train_AUC-ROC": epoch_train_aucROC,
            "val_loss": val_loss,
            "val_accuracy": epoch_val_accuracy,	
            "val_MCC": epoch_val_mcc,
            "val_AUC-ROC": epoch_val_aucROC,
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
    
    custom_dataset_test = CustomDataset(X_test_sequence, X_test_treatment, Y_test)
    data_loader_test = DataLoader(
        custom_dataset_test,
        batch_size=batch_size,
        collate_fn=custom_collate,
    )

    with torch.no_grad():  # Check performance on test set of best model
        model.eval()
        all_predictions = []
        all_probas = []
        all_labels = []
        for batch in tqdm(data_loader_test):
            if only_index:
                batch_X = (
                    torch.tensor(data_npy[batch["X"]], dtype=default_dtype)
                    .to(device)
                    .transpose(1, 2)
                )  # Batch of X sequence (batch, channel, sequence)
            else:
                batch_X = torch.tensor(batch["X"], dtype=default_dtype).to(device)  # Batch of X sequence
            
            batch_Z = torch.tensor(batch["Z"], dtype=default_dtype).to(device)  # Batch of X condition

            outputs = model(batch_X, batch_Z)
            all_predictions.append(torch.max(outputs, 1)[1])  # Get class with highest score
            all_probas.append(outputs)  # Store output probabilities
            batch_Y = torch.tensor(batch["Y"], dtype=torch.long).to(device)
            all_labels.append(batch_Y)
        print("Testing done!")
        
        loss = criterion(torch.cat(all_probas, dim=0), torch.cat(all_labels, dim=0))

        # Concatenate all predictions and probabilities
        all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
        all_predictions = torch.cat(all_predictions, dim=0)
        all_probas = torch.cat(all_probas, dim=0)
        probabilities = torch.nn.functional.softmax(all_probas, dim=1).detach().cpu().numpy()

        # Calculate accuracy
        accuracy = np.mean(all_predictions.detach().cpu().numpy() == all_labels)

        # Calculate AUC-ROC using One-vs-One (OvO)
        auc_scores = roc_auc_score(all_labels, probabilities, multi_class="ovo")
        
        # Plot and save ROC curve
        skplt.metrics.plot_roc(all_labels, probabilities, title="ROC Curve, \nclassification of differentially expressed genes", plot_macro=False, plot_micro=False)
        plt.savefig(f"{path}/{name}_roc_curve.png")

        # Plot and save Precision-Recall Curve
        skplt.metrics.plot_precision_recall(all_labels, probabilities)
        plt.savefig(f"{path}/{name}_precision_recall_curve.png")
        
        # Calculate MCC
        mcc = matthews_corrcoef(all_labels, all_predictions.detach().cpu().numpy())

        print(f"Test Accuracy : {accuracy:.4f}")
        print(f"Test AUC-ROC (OvO): {auc_scores:.4f}")
        print(f"Test MCC : {mcc:.4f}")
        print(f"Test Loss : {loss:.4f}")

        # Save test results
        with open(f"{path}/{name}_test_results.txt", "w") as f:
            f.write(f"Test Accuracy : {accuracy:.4f}\n")
            f.write(f"Test AUC-ROC (OvO) : {auc_scores:.4f}\n")
            f.write(f"Test MCC : {mcc:.4f}\n")
            f.write(f"Test Loss : {loss:.4f}\n")

        # Confusion Matrix
        skplt.metrics.plot_confusion_matrix(all_labels, all_predictions.detach().cpu().numpy(), normalize=False)
        plt.savefig(f"{path}/{name}_confusion_matrix.png")

        # Optional: Calibration plot if needed
        # skplt.metrics.plot_calibration_curve(
        #     Y_test,
        #     [probabilities],  # List of probability predictions for each classifier (if comparing multiple models, extend this list)
        #     clf_names=[name]  # Names of the classifiers
        # )
        # plt.savefig(f"{path}/{name}_calibration_curve.png")

    # Log and plot metrics during training
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 5))

    # Training metrics
    ax[0, 0].plot([epoch_logging[epoch]["train_loss"] for epoch in epoch_logging])
    ax[0, 0].set_title("Train Loss")
    ax[0, 1].plot([epoch_logging[epoch]["train_accuracy"] for epoch in epoch_logging])
    ax[0, 1].set_title("Train Accuracy")
    ax[0, 2].plot([epoch_logging[epoch]["train_MCC"] for epoch in epoch_logging])
    ax[0, 2].set_title("Train MCC")
    ax[0, 3].plot([epoch_logging[epoch]["train_AUC-ROC"] for epoch in epoch_logging])
    ax[0, 3].set_title("Train AUC-ROC")

    # Validation metrics
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


train(
    model,
    X_train_sequence,
    X_train_treatment,
    Y_train,
    X_val_sequence,
    X_val_treatment,
    Y_val,
    X_test_sequence,
    X_test_treatment,
    Y_test,
    device,
    "Results/All_together/CNN/FunProse/thesis_models/dropout_times_0.8_family",
    f"family_splitting_dropout_times0.8_{i}",
    200,
    1024,
    0.00010229218879330196,
    0.0016447149582678627,
    only_index=False,
    #data_npy=np.load(
    #    "Data/Processed/Embeddings/embeddings_agro_nt_full_2024-03-18_13-03-59.npy"
    )
