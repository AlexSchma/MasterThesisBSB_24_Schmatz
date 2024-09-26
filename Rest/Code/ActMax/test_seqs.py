import torch
import numpy as np
from Analysis_Utils import *
from Data_Utils import *
from FunProseModels import *
import os
import json

import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def create_plots(main_losses, adv_losses, results_path, cond):
    # Convert tensors to numpy arrays
    main_losses_np = main_losses.detach().cpu().numpy()
    
    # Prepare arrays for the scatter plot
    prob_cond_overexpressed = main_losses_np[:, 2]  # Third column of main losses tensor
    
    # Calculate the average probabilities for adv losses
    avg_prob_adv_no_change = []
    adv_losses_np = {}
    for key, tensor in adv_losses.items():
        adv_losses_np[key] = tensor.detach().cpu().numpy()
        avg_prob_adv_no_change.append(adv_losses_np[key][:, 1])  # Second column averaged
    
    avg_prob_adv_no_change = np.array(avg_prob_adv_no_change).mean(axis=0)
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(prob_cond_overexpressed, avg_prob_adv_no_change)
    plt.scatter([1], [1], color='red')  # Add point at (1, 1)
    plt.text(0.98, 1, 'Perfect Classification', color='red', fontsize=12, ha='right')
    plt.xlabel(f'Probability {cond} overexpressed')
    plt.ylabel('Avg probability no expression change for other conditions')
    plt.title(f'Overexpression Probability ({cond}) vs. Avg No Change Probability ')
    scatter_plot_path = f"{results_path}/{cond}_scatter_plot.png"
    plt.savefig(scatter_plot_path)
    plt.close()

    # Box plots
    plt.figure(figsize=(12, 10))
    data = [
        main_losses_np[:, 0],  # cond underexpressed
        main_losses_np[:, 1],  # cond no change
        main_losses_np[:, 2],  # cond overexpressed
    ]
    labels = [cond]  # Single label for cond
    
    for key, values in adv_losses_np.items():
        data.extend([values[:, 0], values[:, 1], values[:, 2]])  # Append all three conditions for adv
        labels.extend([key] * 3)  # Single label for adv cond, repeated three times

    box = plt.boxplot(data, patch_artist=True)
    
    # Add colors to boxplot
    colors = ['lightblue', 'lightgreen', 'lightcoral'] * (len(data) // 3)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in ['lightblue', 'lightgreen', 'lightcoral']]
    plt.legend(handles, ['Underexpressed', 'No Change', 'Overexpressed'], loc='upper right')
    
    plt.xticks(ticks=np.arange(2, len(labels) + 2, 3), labels=labels[::3])  # Center labels on "No Change"
    plt.ylabel('Probability')
    plt.title('Boxplots of Probabilities for Different Conditions')
    boxplot_path = f"{results_path}/{cond}_boxplots.png"
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()

    print(f"Scatter plot saved to: {scatter_plot_path}")
    print(f"Boxplots saved to: {boxplot_path}")

def create_combined_confusion_matrix(main_losses, adv_losses, results_path, cond):
    # Convert tensors to numpy arrays
    main_losses_np = main_losses.detach().cpu().numpy()
    
    # Calculate predicted classes for main losses
    main_preds = np.argmax(main_losses_np, axis=1)

    # Calculate predicted classes for adv losses
    all_preds = main_preds.tolist()
    for key, tensor in adv_losses.items():
        adv_losses_np = tensor.detach().cpu().numpy()
        adv_preds = np.argmax(adv_losses_np, axis=1)
        all_preds.extend(adv_preds.tolist())

    # Create true labels (one overexpressed, four no change) for each set of 5 predictions
    num_sets = len(all_preds) // 5
    num_adv_conditions = len(adv_losses)
    true_labels = [2]*num_sets
    true_labels.extend([1]*num_sets*num_adv_conditions)
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, all_preds, labels=[0, 1, 2])

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Underexpressed", "No Change", "Overexpressed"])
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix for {cond}')
    cm_path = f"{results_path}/{cond}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")

def save_losses(main_losses,adv_losses, results_path, cond):
    # Prepare the dictionary structure
    data = {
        "adv_losses":adv_losses ,
        "main_losses": main_losses}
    
    # Save as a JSON file
    save_path = os.path.join(results_path, f"{cond}_losses.json")
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def get_predictions(model, model_dict_path, adv_cond_tensor_dict, iteration, cond, results_path, seq_array, device, lr, cond_tensor):
    model.to(device)
    print("Model created")
    model_state_dict = torch.load(model_dict_path)
    print("Model state dict loaded")
    model.load_state_dict(model_state_dict)
    model.to(device)
    print("Model loaded from checkpoint")
    model.eval()

    input_tensor = torch.tensor(seq_array, requires_grad=False, device=device, dtype=torch.float32)
    dataset = TensorDataset(input_tensor, cond_tensor)
    dataloader = DataLoader(dataset, batch_size=len(seq_array), shuffle=False)
    main_losses = None
    adv_losses = {k: None for k in adv_cond_tensor_dict.keys()}
    
    
    for i,(test_seqs,cond_tensor) in enumerate(dataloader):
        # Main condition tensor
        if i ==1:
            print("Sanity failed, cond tensor looped twice")
        output = model(test_seqs, cond_tensor)
        output = torch.softmax(output, dim=1)
        main_losses = output.detach().cpu()
        # Additional condition tensors
    
    for adv_cond, tensor in adv_cond_tensor_dict.items():
        dataset = TensorDataset(input_tensor, tensor)
        dataloader = DataLoader(dataset, batch_size=len(seq_array), shuffle=False)
        for i,(test_seqs, tensor) in enumerate(dataloader):
            if i ==1:
                print(f"Sanity failed, adv_cond {adv_cond} tensor looped twice")
            adv_output = model(test_seqs, tensor)
            adv_output = torch.softmax(adv_output, dim=1)
            adv_losses[adv_cond] = adv_output.detach().cpu()
              # Continue to accumulate gradients

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    create_plots(main_losses, adv_losses, results_path,cond)
    create_combined_confusion_matrix(main_losses, adv_losses, results_path, cond)

    #save_losses(main_losses, adv_losses, results_path, cond)

    return main_losses, adv_losses


if __name__ == '__main__':
    loaded_sequences = np.load('test_seqs/one_up_seqs.npz')
    loaded_sequences_dict = {key: loaded_sequences[key] for key in loaded_sequences}
    print("Imported all necessary modules")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = get_funprose_multi_class()
    print("Model created")

    #Dictionary of condition embeddings
    cond_tensor_dict = {
        "SA": [0, 1, 0],
        "ABA": [1, 0, 0],
        "SAJA": [0, 1, 1],
        "JA": [0, 0, 1],
        "ABAJA": [1, 0, 1]        
    }
    nuc_seq_dict = {}
    experiment_name = "test_one_up_seqs"
    results_path = f"test_seqs/{experiment_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_losses_accum = None
    adv_losses_mean = None
    adv_losses_appended = None
    index_dict = {}
    # Iterate over conditions and embeddings
    for i, (cond, test_seqs) in enumerate(loaded_sequences.items()):
        # Create the primary condition tensor
        cond = cond.split("up")[0].split("_")[-2]
        embedding= cond_tensor_dict[cond]
        n_sequences = test_seqs.shape[0]
        cond_tensor = torch.tensor([embedding] * n_sequences, device=device).float()

        # Determine alternative conditions by excluding the current condition
        alt_conds = {k: v for k, v in cond_tensor_dict.items() if k != cond}
        
        
        # List to hold alternative condition tensors
        adv_cond_tensor_dict = {}

        # Generate alternative condition tensors
        for alt_cond, alt_embedding in alt_conds.items():
            adv_cond_tensor = torch.tensor([alt_embedding] * n_sequences, device=device).float()
            adv_cond_tensor_dict[alt_cond] = adv_cond_tensor

        main_losses,adv_losses = get_predictions(model=model,
                                                   model_dict_path="models/up_down_multi_class_model_best_model.pth",
                                                   adv_cond_tensor_dict=adv_cond_tensor_dict,
                                                   iteration=i,
                                                   cond=cond,
                                                   results_path=results_path,
                                                   seq_array=test_seqs,
                                                   device=device,
                                                   lr=0.1,
                                                   cond_tensor=cond_tensor,
                                                   )
        if main_losses_accum is None:
            main_losses_accum = main_losses.detach().cpu().numpy()
        else:
            main_losses_accum = np.concatenate((main_losses_accum, main_losses.detach().cpu().numpy()), axis=0)
        index_dict[cond] = len(main_losses_accum)
        if adv_losses_mean is None:
            avg_prob_adv_no_change = []
            adv_losses_np = {}
            for key, tensor in adv_losses.items():
                adv_losses_np[key] = tensor.detach().cpu().numpy()
                avg_prob_adv_no_change.append(adv_losses_np[key][:, 1])  # Second column averaged
            
            adv_losses_mean = np.array(avg_prob_adv_no_change).mean(axis=0)
        else:
            avg_prob_adv_no_change = []
            adv_losses_np = {}
            for key, tensor in adv_losses.items():
                adv_losses_np[key] = tensor.detach().cpu().numpy()
                avg_prob_adv_no_change.append(adv_losses_np[key][:, 1])
            adv_losses_mean = np.concatenate((adv_losses_mean, np.array(avg_prob_adv_no_change).mean(axis=0)), axis=0)
        
        
        for key, tensor in adv_losses.items():
            adv_losses_np = tensor.detach().cpu().numpy()
            adv_preds = np.argmax(adv_losses_np, axis=1)
            if adv_losses_appended is None:
                adv_losses_appended = adv_preds.tolist()
            else:
                adv_losses_appended.extend(adv_preds.tolist())
    main_losses_np = main_losses_accum

    # Prepare arrays for the scatter plot
    prob_cond_overexpressed = main_losses_np[:, 2]  # Third column of main losses tensor

    # Scatter plot with colors based on condition
    plt.figure(figsize=(8, 6))

    # Generate a color map with a different color for each condition
    conditions = list(index_dict.keys())
    colors = plt.colormaps['tab10']

    # Plot each condition with different colors
    start_idx = 0
    for i, cond in enumerate(conditions):
        length = index_dict[cond]
        end_idx = start_idx + length
        plt.scatter(prob_cond_overexpressed[start_idx:end_idx], adv_losses_mean[start_idx:end_idx], color=colors(i / len(conditions)), s=10, label=cond)
        start_idx = end_idx

    plt.scatter([1], [1], color='red', s=50)  # Add point at (1, 1)
    plt.text(0.95, 1, 'Perfect Classification', color='red', fontsize=12, ha='right')

    plt.xlabel('Probability overexpressed')
    plt.ylabel('Avg probability no expression change for other conditions')
    plt.title('Overexpression Probability vs. Avg No Change Probability')
    plt.legend(title='Condition')
    scatter_plot_path = f"{results_path}/combined_scatter_plot.png"
    plt.savefig(scatter_plot_path)
    plt.close()

    print(f"Combined scatter plot saved to: {scatter_plot_path}")

    main_losses_np = main_losses_accum
    
    # Calculate predicted classes for main losses
    main_preds = np.argmax(main_losses_np, axis=1)
    all_preds = np.concatenate([main_preds, adv_losses_appended]).tolist()

    # Create true labels (one overexpressed, four no change) for each set of 5 predictions
    n_ground_truth_true = len(main_preds)
    num_adv_conditions = len(adv_losses)
    true_labels = [2]*n_ground_truth_true
    true_labels.extend([1]*n_ground_truth_true*num_adv_conditions)

    # Combine all predictions
    
    
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, all_preds, labels=[0, 1, 2])

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Underexpressed", "No Change", "Overexpressed"])
    disp.plot(ax=ax)
    plt.title(f'Combined Confusion Matrix')
    cm_path = f"{results_path}/combined_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")