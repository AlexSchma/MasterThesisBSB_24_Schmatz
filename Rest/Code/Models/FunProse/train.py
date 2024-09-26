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
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt
import os
from Code.Models.CNNs._resnet import get_cnn
from Code.Models.CNNs.funprose_adapted import get_funprose

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
default_dtype = torch.float32


# Load XY format one-hot encoded data
with open("Data/Processed/XY_formats/One_hot_encoded.pkl", "rb") as f:
    xy_dict = pickle.load(f)

# load the gene names
gene_names = np.load("Data/Processed/XY_formats/gene_names_one_hot.npy")

np.random.seed(12345)
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

# train_index, test_index = family_wise_train_test_splitting(X_treatment, Y, np.array(family_names)[random_index], random_state = 7215, return_index=True)
train_index, test_index = family_wise_train_test_splitting(
    X_treatment, Y, np.array(family_names), random_state=1, return_index=True
)
# Get valiadation set
train_index, val_index = family_wise_train_test_splitting(
    X_treatment[train_index],
    Y[train_index],
    np.array(family_names)[train_index],
    random_state=2,
    return_index=True,
    test_size=0.3,
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


# model = get_cnn(False, device)
# model = get_cnn(False, device)
model = get_funprose().to(device)
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
        shuffle=True,
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
            optimizer.zero_grad()
            outputs = model(batch_X, batch_Z)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            MCCs.append(
                matthews_corrcoef(
                    batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy()
                )
            )
            accuracies.append((predicted == batch_Y).sum().item() / batch_Y.size(0))
            losses.append(loss.item())
            aucROC.append(
                roc_auc_score(
                    batch_Y.detach().cpu().numpy(), outputs.detach().cpu().numpy()[:, 1]
                )
            )

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
            pbar = tqdm(data_loader_val, total=len(data_loader_val))
            accuracies = []
            losses = []
            MCCs = []
            aucROC = []
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
                outputs = model(batch_X, batch_Z)
                loss = criterion(outputs, batch_Y)
                _, predicted = torch.max(outputs, 1)

                MCCs.append(
                    matthews_corrcoef(
                        batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy()
                    )
                )
                MCCs.append(
                    matthews_corrcoef(
                        batch_Y.detach().cpu().numpy(), predicted.detach().cpu().numpy()
                    )
                )
                accuracies.append((predicted == batch_Y).sum().item() / batch_Y.size(0))
                losses.append(loss.item())
                aucROC.append(
                    roc_auc_score(
                        batch_Y.detach().cpu().numpy(),
                        outputs.detach().cpu().numpy()[:, 1],
                    )
                )
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
    custom_dataset_test = CustomDataset(
                    X_test_sequence, X_test_treatment, Y_test
                )
    data_loader_test = DataLoader(
                    custom_dataset_test,
                    batch_size=batch_size,
                    collate_fn=custom_collate,
                )
    with torch.no_grad():  # Check performance test set of best model
        model.eval()
        all_predictions = []
        all_probas = []
        for batch in tqdm(data_loader_test):
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
            # batch_Y = torch.tensor(batch['Y'], dtype = torch.long).to(device)  # Batch of label

            outputs = model(batch_X, batch_Z)
            _, predicted = torch.max(outputs, 1)
            all_predictions.append(predicted)
            all_probas.append(outputs)
        print("Testing done!")
        all_predictions = torch.cat(all_predictions, dim=0)
        
        all_probas = torch.cat(all_probas, dim=0)
        accuracy = np.mean(all_predictions.detach().cpu().numpy() == Y_test)
        # Calculate AUC-ROC
        auc = roc_auc_score(Y_test, all_probas.detach().cpu().numpy()[:, 1])
        # print(f"AUC-ROC: {auc:.4f}")
        skplt.metrics.plot_roc(Y_test, all_probas.detach().cpu().numpy(), title=f"ROC Curve, \nclassification of differentially expressed genes", plot_macro=False, plot_micro=False)
        plt.savefig(f"{path}/{name}_roc_curve.png")

        #Now precision-recall curve
        skplt.metrics.plot_precision_recall(Y_test, all_probas.detach().cpu().numpy())
        plt.savefig(f"{path}/{name}_precision_recall_curve.png")
        # Calculate MCC
        mcc = matthews_corrcoef(
            Y_test, all_predictions.detach().cpu().numpy()
        )
        print(f"Test MCC : {mcc:.4f}")
        print(f"Test AUC-ROC : {auc:.4f}")
        print(f"Test Accuracy : {accuracy:.4f}")
        # Print confussion matrix
        skplt.metrics.plot_confusion_matrix(Y_test, all_predictions.detach().cpu().numpy(), normalize=False)
        # Remove the color bar
        plt.savefig(f"{path}/{name}_confussion_matrix.png")

        # Calibration plot
        skplt.metrics.plot_calibration_curve(Y_test, [all_probas.detach().cpu().numpy()], clf_names=[name])
        plt.savefig(f"{path}/{name}_calibration_curve.png")

        if num_epochs == 0:
            return accuracy, mcc, auc
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
    "Code/Models/CNNs",
    "FunProse",
    1000,
    128,
    0.00010229218879330196,
    0.0016447149582678627,
    only_index=False,
    #data_npy=np.load(
    #    "Data/Processed/Embeddings/embeddings_agro_nt_full_2024-03-18_13-03-59.npy"
    )
