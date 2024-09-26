import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import chi2_contingency
import pandas as pd
import seaborn as sns
import sys
import os
from torch.utils.data import TensorDataset, DataLoader


from Analysis_Utils import *
from Data_Utils import *
print("Imported all necessary modules")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
def create_synth_test_data_uncond(motifs = ["TTCCCTCG", "CTACCTCC", "TGCAGTGC"],n_synth = 6000, batch_size = 128):
     
    sequences, labels = zip(*(generate_sequence(motifs=motifs) for _ in range(n_synth)))  
    one_hot_sequences = one_hot_encode(sequences)


    X_test = np.array(one_hot_sequences)
    y_test = np.array(labels)
    # Convert to PyTorch tensors
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    # Create DataLoaders
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return test_loader, X_test, y_test

def evaluate_model_uncond(model, test_loader):
    model.eval()
    test_outputs, test_labels = [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            output = model(sequences).squeeze()
            test_outputs.append(output.sigmoid())
            test_labels.append(labels)

    test_outputs = torch.cat(test_outputs).cpu().numpy()
    test_labels = torch.cat(test_labels).cpu().numpy()

    fpr, tpr, thresholds = roc_curve(test_labels, test_outputs)
    roc_auc = auc(fpr, tpr)

    accuracy = (test_outputs.round() == test_labels).mean()
    
    return fpr, tpr, roc_auc, accuracy

# def plot_roc_curves(roc_data):
#     plt.figure()
#     for model_name, (fpr, tpr, roc_auc) in roc_data.items():
#         plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves Comparison')
#     plt.legend(loc="lower right")
#     if _6mer:
#         plt.savefig(f'{results_dir}/uncond_6merMotif_roc_curves.png')
#     else:
#         plt.savefig(f'{results_dir}/uncond_roc_curves.png')
#     plt.close()

# def plot_accuracy_bar(accuracies):
#     plt.figure()
#     sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
#     plt.title('Model Accuracy Comparison')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Model')
#     if _6mer:
#         plt.savefig(f'{results_dir}/uncond_6merMotif_accuracy_bar.png')
#     else:

#         plt.savefig(f'{results_dir}/uncond_accuracy_bar.png')
#     plt.close()

def average_motif_occurrences_uncond(motifs, nuc_seqs_per_model):
    avg_occurrences = pd.DataFrame(index=motifs)

    for model_name, nuc_seqs in nuc_seqs_per_model.items():
        total_counts = {motif: 0 for motif in motifs}
        for seq in nuc_seqs:
            for motif in motifs:
                total_counts[motif] += seq.count(motif)
        avg_occurrences[model_name] = [count / len(nuc_seqs) for count in total_counts.values()]

    return avg_occurrences

def plot_motif_occurrences_uncond(avg_occurrences,results_dir,Basemodel_name):
    avg_occurrences.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Motif Occurrences per Sequence per Model')
    plt.ylabel('Average Occurrences')
    plt.xlabel('Motif')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plot_filename = f'{Basemodel_name}_motif_occurrences.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()

def generate_nuc_seqs_uncond(model, device, X_train_shape, batch_size=100, lr=0.05, max_iters=1000, stop_threshold=10):
    model.eval()

    # Define an optimizable input tensor, starting with random noise
    input_tensor = torch.randn((batch_size, X_train_shape[1], 4), requires_grad=True, device=device)

    # Define an optimizer for the input tensor
    optimizer = torch.optim.Adam([input_tensor], lr=lr)

    sequence_history = []  # To track the sequences and check for early stopping

    for i in range(max_iters):
        optimizer.zero_grad()

        # Forward pass
        output = model(input_tensor).squeeze()

        # Maximize the output using negative loss
        loss = -torch.mean(output)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Convert the optimized input tensor to DNA sequences
        argmax_optimized_input = torch.argmax(input_tensor, dim=2).cpu().numpy()
        nuc_seqs = argmax_to_nucleotide(argmax_optimized_input)  # Assume this function is defined elsewhere

        # Early stopping check
        sequence_history.append(nuc_seqs)
        if len(sequence_history) > stop_threshold:
            sequence_history.pop(0)

            # Check if the sequences have not changed over the last few iterations
            if all(np.array_equal(seq, sequence_history[-1]) for seq in sequence_history[:-1]):
                print(f"Early stopping at iteration {i} due to no change in DNA sequences over the last iterations.")
                break

    return sequence_history[-1]  # Return the last set of sequences

def run_model_analysis_uncond(_6mer,motifs,Base_model_name,n_models,device='cuda'):


    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Load and evaluate models
    if _6mer:
        Base_model_name = f'{Base_model_name}_6merMotifs'
    
    model_names = [f'{Base_model_name}_{i}' for i in range(n_models)]
    roc_data, accuracies, nuc_seqs_per_model = {}, {}, {}
    test_loader,X_test,y_test = create_synth_test_data_uncond(motifs=motifs)
    print("Created test data")
    for model_name in model_names:
        model = torch.load(f'models/{model_name}.pth').to(device)
        fpr, tpr, roc_auc, accuracy = evaluate_model_uncond(model, test_loader)
        roc_data[model_name] = (fpr, tpr, roc_auc)
        accuracies[model_name] = accuracy
        print(f"Model {model_name} evaluated with accuracy {accuracy:.2f} and AUC {roc_auc:.2f}")
        nuc_seqs_per_model[model_name] = generate_nuc_seqs_uncond(model, device, X_test.shape, batch_size=100)
        print(f"Generated nucleotide sequences for model {model_name}")
    # Visualization
    plot_roc_curves(Base_model_name,roc_data,results_dir)
    plot_accuracy_bar(Base_model_name,accuracies,results_dir)
    print("Plotted ROC curves and accuracy bar")
    # Motif analysis and visualization
    results = average_motif_occurrences_uncond(motifs, nuc_seqs_per_model)
    plot_motif_occurrences_uncond(results,results_dir,Base_model_name)
    print("Plotted motif occurrences")
if __name__ == "__main__":
    run_model_analysis_uncond(_6mer=True,motifs=["TTCCCT", "CTACCT", "TGCAGT"],Base_model_name="BigFilterMaxPool",n_models=2,device='cuda')