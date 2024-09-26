import random
# import sys
# sys.path.append("./")
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import chi2_contingency
import os
from Data_Utils import *
from Models import *
from Analysis_Utils import *



def train_and_save( train_loader, val_loader,epochs=150,n_models=5,lr=0.001,L2=1e-5,_6mer=False,dropout_rate=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if not os.path.exists('models'):
    # If not, create it
        os.makedirs('models')

    for model_idx in range(n_models):
        model = ModifiedSimpleCNNCondOnConvMultiply(dropout_rate=dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=L2)
        
        # Initialize TensorBoard SummaryWriter
        writer = SummaryWriter()
        
        # Initialize metrics
        accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)
        auroc_metric = torchmetrics.AUROC(task="binary").to(device)
        
        
        for epoch in range(epochs):
            model.train()
            for sequences, labels, conditions in train_loader:
                sequences, labels, conditions = sequences.to(device), labels.to(device), conditions.to(device)
                optimizer.zero_grad()
                output = model(sequences,conditions).squeeze()
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            
            # Validation loop
            model.eval()
            with torch.no_grad():
                val_loss = 0
                all_outputs = []
                all_labels = []
                for sequences, labels, conditions in val_loader:
                    sequences, labels, conditions = sequences.to(device), labels.to(device), conditions.to(device)
                    output = model(sequences,conditions).squeeze()
                    val_loss += criterion(output, labels).item()
                    
                    # Store outputs and labels to calculate AUROC
                    all_outputs.append(output)
                    all_labels.append(labels)
        
                    # Calculate accuracy for current batch
                    accuracy_metric(output.sigmoid() >= 0.5, labels.int())
                
                # Calculate overall accuracy for this epoch
                accuracy = accuracy_metric.compute()
        
                # Reset accuracy metric for next epoch
                accuracy_metric.reset()
        
                # Concatenate all outputs and labels from all batches
                all_outputs = torch.cat(all_outputs)
                all_labels = torch.cat(all_labels)
                
                # Calculate AUROC for this epoch
                auroc = auroc_metric(all_outputs.sigmoid(), all_labels.int())
                
                # Reset AUROC metric for next epoch
                auroc_metric.reset()
        
                # Log metrics to TensorBoard
                writer.add_scalar("Loss/train", loss.item(), epoch)
                writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
                writer.add_scalar("Accuracy/val", accuracy.item(), epoch)
                writer.add_scalar("AUROC/val", auroc.item(), epoch)
        
                print(f"Model {model_idx}, Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {accuracy.item()}, Val AUROC: {auroc.item()}")
        
        # Close the TensorBoard SummaryWriter
        writer.close()
        if _6mer:
        
            torch.save(model, f'models/BigFilterMaxPoolCondEmbeddingMultiply_6merMotifs_{model_idx}.pth')
            print(f"Model {model_idx} saved to BigFilterMaxPoolCondEmbeddingMultiply_6merMotifs_{model_idx}.pth")
        else:
            torch.save(model, f'models/BigFilterMaxPoolCondEmbeddingMultiply_{model_idx}.pth')
            print(f"Model {model_idx} saved to BigFilterMaxPoolCondEmbeddingMultiply_{model_idx}.pth")

if __name__ == "__main__":
    _6mer = True
    if _6mer:
        motifs = ["TTCCCT", "CTACCT", "TGCAGT"]
    else:
        motifs = ["TTCCCTCG", "CTACCTCC", "TGCAGTGC"]
    train_loader, val_loader, test_loader = create_synth_data(motifs=motifs)
    train_and_save(train_loader, val_loader,epochs=70,n_models=5,lr=0.001,L2=1e-5,_6mer=_6mer,dropout_rate=0.2)
    