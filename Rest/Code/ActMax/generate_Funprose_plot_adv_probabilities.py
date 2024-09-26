import torch
import numpy as np
from Analysis_Utils import *
from Data_Utils import *
from FunProseModels import *
import os
import json

import matplotlib.pyplot as plt
import torch.nn.functional as F


def instance_normalize(logits):
    mean = logits.mean(dim=2, keepdim=True)
    std = logits.std(dim=2, keepdim=True)
    return (logits - mean) / (std + 1e-5)

def reduce_parameter(initial_param, iteration, max_iters,end_param):
    return initial_param - (initial_param - end_param) * (iteration / max_iters)

def calculate_relative_hamming_distance(previous, current):
    prev_indices = previous.argmax(dim=1)
    curr_indices = current.argmax(dim=1)
    differences = (prev_indices != curr_indices).float()  
    relative_distances = differences.sum(dim=-1) / differences.shape[-1]  
    return relative_distances


def entropy_loss_func(pwm):
    pwm = torch.clamp(pwm, min=1e-9, max=1 - 1e-9)
    entropy = -pwm * torch.log2(pwm)
    entropy = entropy.sum(dim=1)
    mean_entropy = entropy.mean(dim=1)
    return mean_entropy.mean()



def target_entropy_mse(pwm, target_bits):
    pwm_clipped = torch.clamp(pwm, min=1e-8, max=1.0 - 1e-8)
    entropy = pwm_clipped * -torch.log(pwm_clipped) / torch.log(torch.tensor(2.0))
    entropy = torch.sum(entropy, dim=1)
    conservation = 2.0 - entropy
    mse = torch.mean((conservation - target_bits)**2)
    return mse

def visualize_and_save(losses, hamming_distances, results_path,cond,show=False):
    save_path = os.path.join(results_path, f"{cond}_loss_vs_hamming.png")

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(len(losses)), losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average Relative Hamming Distance', color=color)
    ax2.plot(range(len(hamming_distances)), hamming_distances, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Loss and Average Relative Hamming Distance over Iterations')
    plt.savefig(save_path)
    if show:
        plt.show()

def plot_adv_results(results_adv_dict, cond,results_path):
    """
    Plots the adversarial results saved in results_adv_dict.

    :param results_adv_dict: Dictionary where keys are alternative conditions,
                             and values are tensors with three elements
                             representing probabilities of being underexpressed, no change, or overexpressed.
    :param cond: The current condition for the title.
    """
    save_path = os.path.join(results_path, f"{cond}_adv_probabilities.png")

    # Extract data from the dictionary
    labels = list(results_adv_dict.keys())
    data = np.array([results_adv_dict[label] for label in labels])

    # Split the data into separate lists for plotting
    underexpressed = data[:, 0]
    no_change = data[:, 1]
    overexpressed = data[:, 2]

    # Create a stacked bar plot
    bar_width = 0.6
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))

    plt.bar(x, underexpressed, bar_width, label="Underexpressed", color='blue')
    plt.bar(x, no_change, bar_width, label="No Change", bottom=underexpressed, color='gray')
    plt.bar(x, overexpressed, bar_width, label="Overexpressed", bottom=underexpressed + no_change, color='green')

    # Adding labels and titles
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Alternative Conditions")
    plt.ylabel("Probability")
    plt.title(f"Adversarial Conditions for {cond}")

    plt.legend(loc='upper left')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.savefig(save_path)

def train(model, model_dict_path,adv_cond_tensor_dict,iteration,cond, results_path, input_tensor_shape, device, lr, cond_tensor, initial_tau, final_tau, initial_entropy_weight, final_entropy_weight, target_bits, max_iters, verbose, change_tau, use_custom_entropy_loss, use_fast_seq_entropy_loss):
    model.to(device)
    print("Model created")
    model_state_dict = torch.load(model_dict_path)
    print("Model state dict loaded")
    model.load_state_dict(model_state_dict)
    model.to(device)
    print("Model loaded from checkpoint")
    model.eval()

    input_tensor = torch.randn(input_tensor_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([input_tensor], lr=lr)

    losses = []
    hamming_distances = []
    count = 0
    sample_cache = torch.zeros(input_tensor_shape, device=device)

    assert not (use_custom_entropy_loss and use_fast_seq_entropy_loss), "Choose only one entropy loss function"

    for i in range(max_iters):
        optimizer.zero_grad()
        if change_tau:
            tau = reduce_parameter(initial_tau, i, max_iters, final_tau)
        else:
            tau = initial_tau
        normalized_logits = instance_normalize(input_tensor)
        pwm = F.softmax(normalized_logits, dim=1)
        samples = F.gumbel_softmax(normalized_logits, tau=tau, hard=True, dim=1)
        if i > 0:
            hamming_distance = calculate_relative_hamming_distance(sample_cache, samples)
            hamming_distances.append(hamming_distance.mean().item())
        if torch.allclose(samples, sample_cache):
            count += 1
            if count == 10:
                break
        else:
            count = 0
            sample_cache = samples.clone()
        
        output = model(samples, cond_tensor)
        output = torch.softmax(output, dim=1)
        target_loss = -torch.mean(output[:, 2])

        if use_custom_entropy_loss:
            entropy_weight = reduce_parameter(initial_entropy_weight, i, max_iters, final_entropy_weight)
            loss = entropy_weight * entropy_loss_func(pwm) + target_loss
        elif use_fast_seq_entropy_loss:
            entropy_loss = target_entropy_mse(pwm, target_bits=target_bits)
            loss = target_loss + entropy_loss
        else:
            loss = target_loss
        
        if verbose and i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
            print(f"Output: {output.mean(dim=0)}")
        if i > 0:
            losses.append(loss.item())

        loss.backward()
        optimizer.step()
    results_adv_dict = {}
    for alt_cond, adv_cond_tensor in adv_cond_tensor_dict.items():
        alt_output = model(samples, adv_cond_tensor)
        alt_output = torch.softmax(alt_output, dim=1)
        results_adv_dict[alt_cond] = alt_output.mean(dim=0).detach().cpu().numpy()
    
    
    

    optimized_inputs = input_tensor.detach().cpu()
    argmax_optimized_input = np.argmax(optimized_inputs, 1)
    nuc_seqs = argmax_to_nucleotide(argmax_optimized_input)


    if not os.path.exists(results_path):
        os.makedirs(results_path)
    plot_adv_results(results_adv_dict, cond,results_path)
    visualize_and_save(losses, hamming_distances, results_path,cond)
    # Save nuc_seqs as fasta
    fasta_path = os.path.join(results_path, f"{cond}.fasta")
    with open(fasta_path, "w") as fasta_file:
        for i, seq in enumerate(nuc_seqs):
            fasta_file.write(f">Sequence_{i+1}\n")
            fasta_file.write(f"{seq}\n")

    # Save nuc_seqs as txt
    txt_path = os.path.join(results_path, f"{cond}.txt")
    with open(txt_path, "w") as txt_file:
        for i, seq in enumerate(nuc_seqs):
            txt_file.write(f"Sequence {i+1}: {seq}\n")

    # Save nuc_seqs as numpy file
    np_path = os.path.join(results_path, f"{cond}.npy")
    np.save(np_path, nuc_seqs)
    if iteration == 0:
        hyperparameters = {
            "model_dict_path": model_dict_path,
            "results_path": results_path,
            "input_tensor_shape": input_tensor_shape,
            "device": str(device),
            "lr": lr,
            "initial_tau": initial_tau,
            "final_tau": final_tau,
            "initial_entropy_weight": initial_entropy_weight,
            "final_entropy_weight": final_entropy_weight,
            "target_bits": target_bits,
            "max_iters": max_iters,
            "verbose": verbose,
            "change_tau": change_tau,
            "use_custom_entropy_loss": use_custom_entropy_loss,
            "use_fast_seq_entropy_loss": use_fast_seq_entropy_loss
        }

        with open(os.path.join(results_path, "hyperparameters.json"), "w") as json_file:
            json.dump(hyperparameters, json_file)
    return nuc_seqs, losses, hamming_distances


if __name__ == '__main__':
    print("Imported all necessary modules")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = get_funprose_multi_class()
    print("Model created")
    cond_tensor = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]], device=device).float()

    # Dictionary of condition embeddings
    cond_tensor_dict = {
        "ABA": [1, 0, 0],
        "SA": [0, 1, 0],
        "JA": [0, 0, 1],
        "ABA_JA": [1, 0, 1],
        "SA_JA": [0, 1, 1],
    }

    n_sequences = 100
    nuc_seq_dict = {}
    experiment_name = "no_family_splitting_1000prom_4_model_best_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over conditions and embeddings
    for i, (cond, embedding) in enumerate(cond_tensor_dict.items()):
        # Create the primary condition tensor
        cond_tensor = torch.tensor([embedding] * n_sequences, device=device).float()

        # Determine alternative conditions by excluding the current condition
        alt_conds = {k: v for k, v in cond_tensor_dict.items() if k != cond}

        # List to hold alternative condition tensors
        adv_cond_tensor_dict = {}

        # Generate alternative condition tensors
        for alt_cond, alt_embedding in alt_conds.items():
            adv_cond_tensor = torch.tensor([alt_embedding] * n_sequences, device=device).float()
            adv_cond_tensor_dict[alt_cond] = adv_cond_tensor

        nuc_seq_dict[cond], loss, ham_dist = train(model=model,
                                                   model_dict_path="models/up_down_multi_class_model_best_model.pth",
                                                   adv_cond_tensor_dict=adv_cond_tensor_dict,
                                                   iteration=i,
                                                    cond=cond,
                                                    results_path=f"generated_sequences/{experiment_name}",
                                                    input_tensor_shape=(n_sequences,4,1000),
                                                    device=device, 
                                                    lr=0.1, 
                                                    cond_tensor=cond_tensor, 
                                                    initial_tau = 1.0, 
                                                    final_tau = 0.1, 
                                                    initial_entropy_weight = 0.01, 
                                                    final_entropy_weight = 0.001, 
                                                    target_bits = 0.5, 
                                                    max_iters = 300, 
                                                    verbose = True, 
                                                    change_tau = True, 
                                                    use_custom_entropy_loss = False, 
                                                    use_fast_seq_entropy_loss = False)
                                            
    

    

    
