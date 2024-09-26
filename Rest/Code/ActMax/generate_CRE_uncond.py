import torch
import numpy as np
from Analysis_Utils import *
from Data_Utils import *
from FunProseModels import *
import os
import json
from utils import ConvNet
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

def visualize_and_save(losses, hamming_distances, results_path,cond,direction,show=False):
    save_path = os.path.join(results_path, f"{cond}_{direction}_loss_vs_hamming.png")

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

def train(model,direction, model_dict_path,treatment, results_path, input_tensor_shape, device, lr, initial_tau, final_tau, initial_entropy_weight, final_entropy_weight, target_bits, max_iters, verbose, change_tau, use_custom_entropy_loss, use_fast_seq_entropy_loss):
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
    lowest_hamming_distance = 1.0
    hamming_patience = 50
    hamming_patience_counter = 0
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
            if hamming_distance.mean().item() < lowest_hamming_distance:
                lowest_hamming_distance = hamming_distance.mean().item()
                hamming_patience_counter = 0
            else:
                hamming_patience_counter += 1
            if hamming_patience_counter >= hamming_patience:
                print(f"Hamming distance has not improved for {hamming_patience} iterations. Stopping early.")
                break


        if torch.allclose(samples, sample_cache):
            count += 1
            if count == 10:
                break
        else:
            count = 0
            sample_cache = samples.clone()
        
        output = model(samples)
        if direction == "over":
            target_loss = -torch.mean(output)
        elif direction == "under":
            target_loss = torch.mean(output)

        if use_custom_entropy_loss:
            entropy_weight = reduce_parameter(initial_entropy_weight, i, max_iters, final_entropy_weight)
            loss = entropy_weight * entropy_loss_func(pwm) + target_loss
        elif use_fast_seq_entropy_loss:
            entropy_loss = target_entropy_mse(pwm, target_bits=target_bits)
            loss = target_loss + entropy_loss
        else:
            loss = target_loss
        
        if verbose and i % 1 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
            print(f"Output: {output.mean(dim=0)}")
        if i > 0:
            losses.append(loss.item())

        loss.backward()
        optimizer.step()

    optimized_inputs = input_tensor.detach().cpu()
    argmax_optimized_input = np.argmax(optimized_inputs, 1)
    nuc_seqs = argmax_to_nucleotide(argmax_optimized_input)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    visualize_and_save(losses, hamming_distances, results_path,treatment,direction)
    # Save nuc_seqs as fasta
    fasta_path = os.path.join(results_path, f"{treatment}_{direction}.fasta")
    with open(fasta_path, "w") as fasta_file:
        for i, seq in enumerate(nuc_seqs):
            fasta_file.write(f">Sequence_{i+1}\n")
            fasta_file.write(f"{seq}\n")

    # Save nuc_seqs as txt
    txt_path = os.path.join(results_path, f"{treatment}_{direction}.txt")
    with open(txt_path, "w") as txt_file:
        for i, seq in enumerate(nuc_seqs):
            txt_file.write(f"Sequence {i+1}: {seq}\n")

    # Save nuc_seqs as numpy file
    np_path = os.path.join(results_path, f"{treatment}_{direction}.npy")
    np.save(np_path, nuc_seqs)
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
    model = ConvNet().to(device)
    print("Model created")
    n_sequences = 100
    nuc_seq_dict = {}
    dir_base_name = "over_under_100_"
    hormones = ["ABA","JA","ABAJA","SAJA","SA"]
    save_folder_name = "over_under_100_patience50"
    for i in hormones:
        for j in ["over","under"]:
            experiment_name = dir_base_name + i
            treatment = i
            model_dict_path = f"Data/ActMax/{experiment_name}/final/final_model.pth"

            nuc_seq_dict[i], loss, ham_dist = train(model=model,direction=j,
                model_dict_path=model_dict_path,treatment=treatment,
                results_path=f"Data/ActMax/generated_sequences/{save_folder_name}",
                input_tensor_shape=(n_sequences,4,3020),
                device=device, 
                lr=0.1, 
                initial_tau = 1.0, 
                final_tau = 0.1, 
                initial_entropy_weight = 0.01, 
                final_entropy_weight = 0.001, 
                target_bits = 0.5, 
                max_iters = 1000, 
                verbose = True, 
                change_tau = True, 
                use_custom_entropy_loss = False, 
                use_fast_seq_entropy_loss = False)
    
    

    

    
