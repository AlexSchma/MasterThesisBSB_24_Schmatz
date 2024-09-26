import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns
import os
from torch.utils.data import TensorDataset, DataLoader
import pickle
# Assuming the necessary functions are defined in these modules
from Analysis_Utils import *
from Data_Utils import *
print("Imported all necessary modules")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")




def evaluate_model(model, test_loader):
    model.eval()
    test_outputs, test_labels = [], []

    with torch.no_grad():
        for sequences, labels, conditions in test_loader:
            sequences, labels, conditions = sequences.to(device), labels.to(device), conditions.to(device)
            output = model(sequences, conditions).squeeze()
            test_outputs.append(output.sigmoid())
            test_labels.append(labels)

    test_outputs = torch.cat(test_outputs).cpu().numpy()
    test_labels = torch.cat(test_labels).cpu().numpy()

    fpr, tpr, thresholds = roc_curve(test_labels, test_outputs)
    roc_auc = auc(fpr, tpr)

    accuracy = (test_outputs.round() == test_labels).mean()

    return fpr, tpr, roc_auc, accuracy

def generate_nuc_seqs(model, device, X_train_shape, conditions, batch_size, lr=0.05, max_iters=300):
    model.eval()

    input_tensor = torch.randn((batch_size, X_train_shape[1], 4), requires_grad=True, device=device)
    conditions = conditions.float()  # Convert conditions to float

    optimizer = torch.optim.Adam([input_tensor], lr=lr)

    for i in range(max_iters):
        optimizer.zero_grad()

        output = model(input_tensor, conditions).squeeze()

        loss = -torch.mean(output)

        loss.backward()
        optimizer.step()

    optimized_inputs = input_tensor.detach().cpu()
    argmax_optimized_input = np.argmax(optimized_inputs, 2)
    nuc_seqs = argmax_to_nucleotide(argmax_optimized_input)

    return nuc_seqs

# Setting the flag for 6-mer motifs
_6mer = True
        
# Define motifs and conditions based on the _6mer flag
if _6mer:
    motifs = ["TTCCCT", "CTACCT", "TGCAGT"]
    model_names = [f'BigFilterMaxPoolCondEmbedding_6merMotifs_{i}' for i in range(5)]
    Base_model_name = 'BigFilterMaxPoolCondEmbedding_6merMotifs'
else:
    motifs = ["TTCCCTCG", "CTACCTCC", "TGCAGTGC"]
    model_names = [f'BigFilterMaxPoolCondEmbedding_{i}' for i in range(5)]
    Base_model_name = 'BigFilterMaxPoolCondEmbedding'
# Define conditions as tensors for the conditional model
test_conditions = 100
cond_array = [torch.zeros(test_conditions, device=device),
              torch.ones(test_conditions, device=device),
              torch.full((test_conditions,), 2,device=device)]



# Setup results directory
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load models, evaluate, and generate sequences
roc_data, accuracies, nuc_seqs_per_model = {}, {}, {}

test_loader, X_test, y_test, conditions = create_synth_test_data(motifs=motifs)
print("Created synthetic test data")
for model_name in model_names:
    model = torch.load(f'models/{model_name}.pth').to(device)
    fpr, tpr, roc_auc, accuracy = evaluate_model(model, test_loader)
    print(f"Model {model_name} evaluated")
    roc_data[model_name] = (fpr, tpr, roc_auc)
    accuracies[model_name] = accuracy
    nuc_seqs_per_condition = {}
    for condition in cond_array:
        nuc_seqs = generate_nuc_seqs(model, device,X_test.shape , condition,batch_size=test_conditions)
        condition_label = condition[0].item()  # Assuming the condition tensors are uniform
        nuc_seqs_per_condition[str(condition_label)] = nuc_seqs

    nuc_seqs_per_model[model_name] = nuc_seqs_per_condition
    print(f"Nucleotide sequences generated for {model_name}")

#Visualization
plot_roc_curves(Base_model_name,roc_data,results_dir)
plot_accuracy_bar(Base_model_name,accuracies,results_dir)
print("Plots saved to results directory")

# Motif analysis and visualization
results = average_motif_occurrences(motifs, nuc_seqs_per_model)
plot_motif_occurrences(results, motifs, results_dir)
print("Motif occurrences plotted")
#Save the results
# results_filename = "results.pkl"
# with open(os.path.join(results_dir, results_filename), "wb") as f:
#     pickle.dump(results, f)
# print("Results saved as pickle file")
# Load the results from the pickle file
# results_filename = "results.pkl"
# with open(os.path.join(results_dir, results_filename), "rb") as f:
#     results = pickle.load(f)
# print("Results loaded from pickle file")

aggregated_results = aggregate_motif_occurrences(results, motifs)
plot_aggregated_motif_occurrences(Base_model_name,aggregated_results, results_dir)
print("Aggregated motif occurrences plotted")

preference_scores = calculate_preference_scores(results, motifs)
aggregated_scores = aggregate_scores_across_models(preference_scores, motifs)
plot_aggregated_preference_scores(aggregated_scores, results_dir,Base_model_name)
print("Aggregated preference scores plotted")


aggregated_increases = calculate_relative_increases(results, motifs)
plot_relative_increases(aggregated_increases, results_dir,Base_model_name)
print("Relative increases in motif occurrences plotted")

print("All analyses completed and plots saved")

