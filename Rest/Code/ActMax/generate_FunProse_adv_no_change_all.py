import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from FunProseModels import *
torch.set_float32_matmul_precision('high')
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def argmax_to_nucleotide(argmax_sequences):
    argmax_mapping = {0:"A",1:"C",2:"T",3:"G"} #jordi mapping
    nuc_seqs = []
    for argmax_seq in argmax_sequences:
        nuc_seqs.append("".join([argmax_mapping[int(integer)] for integer in argmax_seq]))
    return nuc_seqs

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


def visualize_and_save(adv_losses, total_losses, hamming_distances, results_path, show=False):
    save_path = os.path.join(results_path, f"no_change_loss_vs_hamming.png")

    fig, ax1 = plt.subplots()

    # Plotting Losses
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    
    # Adversarial losses plot
    for key, adv_loss in adv_losses.items():
        ax1.plot(range(len(adv_loss)), adv_loss, label=f'Adv Loss {key}', linestyle=':', alpha=0.7)
    
    # Total loss plot
    ax1.plot(range(len(total_losses)), total_losses, color='tab:orange', label='Total Loss')
    ax1.tick_params(axis='y')
    
    # Adding the second y-axis for Hamming Distance
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Average Relative Hamming Distance', color='tab:brown')
    ax2.plot(range(len(hamming_distances)), hamming_distances, color='tab:brown', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    # Create a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    fig.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig(save_path)
    
    if show:
        plt.show()


def save_losses_and_hamming(losses, hamming_distances, results_path, cond):
    # Prepare the dictionary structure
    data = {
        "total_losses": losses[0],
        "adv_losses": losses[1],  # This can be a dictionary directly
        "main_losses": losses[2],
        "hamming_distances": hamming_distances,
    }
    
    # Save as a JSON file
    save_path = os.path.join(results_path, f"{cond}_losses_hamming.json")
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Loss function: weighted average of expression change loss for all conditions
def weighted_average_loss(model, samples, cond_tensor_dict, pwm, use_custom_entropy_loss, use_fast_seq_entropy_loss, i, max_iters):
    total_loss = 0
    individual_losses = {}
    for cond_name, cond_tensor in cond_tensor_dict.items():
        output = model(samples, cond_tensor)
        output = torch.softmax(output, dim=1)
        cond_loss = -torch.mean(output[:, 1])  # Target for no expression change
        
        # Add entropy-based loss if specified
        if use_custom_entropy_loss:
            entropy_weight = reduce_parameter(initial_entropy_weight, i, max_iters, final_entropy_weight)
            cond_loss += entropy_weight * entropy_loss_func(pwm)
            
        elif use_fast_seq_entropy_loss:
            entropy_loss = target_entropy_mse(pwm, target_bits=target_bits)
            cond_loss += entropy_loss
            
        total_loss += cond_loss
        #print(f"Condition: {cond_name}, Loss: {cond_loss.item()}")
        individual_losses[cond_name] = cond_loss.item()
    return total_loss / len(cond_tensor_dict),individual_losses  # Return the average loss over all conditions

# Helper function to reduce parameter over iterations
def reduce_parameter(initial_value, current_iter, max_iters, final_value):
    return initial_value + (final_value - initial_value) * (current_iter / max_iters)

# Training function with no change in expression optimization
def train(model, model_dict_path, cond_tensor_dict, max_iters, input_tensor_shape, device, lr, initial_tau, final_tau,
          initial_entropy_weight, final_entropy_weight, target_bits, verbose, change_tau, use_custom_entropy_loss,
          use_fast_seq_entropy_loss):

    optimizer = Adam(model.parameters(), lr=lr)
    count = 0
    pbar = tqdm(total=max_iters)
    sample_cache = torch.zeros(input_tensor_shape, device=device)
    # Initializing empty loss tracking lists
    total_losses = []
    hamming_distances = []
    individual_losses_cache = {cond_name: [] for cond_name in cond_tensor_dict.keys()}
    
    for i in range(max_iters):
        optimizer.zero_grad()  # Zero gradients
        
        # Update temperature (tau) if needed
        tau = reduce_parameter(initial_tau, i, max_iters, final_tau) if change_tau else initial_tau

        # Generate samples and calculate PWM
        normalized_logits = torch.randn(input_tensor_shape, device=device, requires_grad=True)
        pwm = F.softmax(normalized_logits, dim=1)
        samples = F.gumbel_softmax(normalized_logits, tau=tau, hard=True, dim=1)
        
        # Cache and early stopping logic
        if i > 0:
            hamming_distance = calculate_relative_hamming_distance(sample_cache, samples)
            hamming_distances.append(hamming_distance.mean().item())

            if torch.allclose(samples, sample_cache):
                count += 1
                if count == 10:  # Stop early if the samples haven't changed for 10 iterations
                    break
            else:
                count = 0
            sample_cache = samples.clone()

        # Compute the weighted average loss for all conditions
        total_loss, individual_losses = weighted_average_loss(model, samples, cond_tensor_dict, pwm, use_custom_entropy_loss, use_fast_seq_entropy_loss, i, max_iters)
        #print(f"Total Loss: {total_loss.item()}")
        #print(f"Individual Losses: {individual_losses}")
        individual_losses_cache = {cond_name: individual_losses_cache[cond_name] + [individual_losses[cond_name]] for cond_name in individual_losses_cache.keys()}
        # Backpropagation and optimization step
        total_loss.backward()
        optimizer.step()

        # Track loss for monitoring
        total_losses.append(total_loss.item())

        if verbose and i % 10 == 0:
            print(f"Iteration {i}, Total Loss: {total_loss.item()}")

        pbar.set_postfix({'Total Loss': total_losses[-1]})
        pbar.update(1)
    
    pbar.close()
    
    # Save the model after training
    torch.save(model.state_dict(), model_dict_path)
    return total_losses, individual_losses_cache, hamming_distances,samples

# Assuming other helper functions like calculate_relative_hamming_distance, entropy_loss_func, etc., are defined

# Example call for training
# Define model, conditions, etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = get_funprose_multi_class()
model.to(device)
model = torch.compile(model)
model_dict_path='models/multiclass_standard_metrics_1_model08do108ep.pth'
experiment_name = 'DO8_108_endmodel'
results_path = f"generated_sequences/{experiment_name}"
os.makedirs(results_path, exist_ok=True)
print("Model created")
model_state_dict = torch.load(model_dict_path)
print("Model state dict loaded")
model.load_state_dict(model_state_dict)
model.to(device)  # Replace with your actual model
cond_dict = {
        "ABA": [1, 0, 0],
        "SA": [0, 1, 0],
        "JA": [0, 0, 1],
        "ABA_JA": [1, 0, 1],
        "SA_JA": [0, 1, 1],
    }
cond_tensor_dict = {}
for alt_cond, alt_embedding in cond_dict.items():
                adv_cond_tensor = torch.tensor([alt_embedding] * 100, device=device).float()
                cond_tensor_dict[alt_cond] = adv_cond_tensor

input_tensor_shape = (100, 4, 4001)  # Example shape
max_iters = 300

# Start training
total_losses, individual_losses_cache, hamming_distances,samples = train(model=model,
      model_dict_path='DO8_108_endmodel.pth',
      cond_tensor_dict=cond_tensor_dict,
      max_iters=max_iters,
      input_tensor_shape=input_tensor_shape,
      device=device,
      lr=0.01,
      initial_tau=1.0,
      final_tau=0.1,
      initial_entropy_weight=0.01,
      final_entropy_weight=0.001,
      target_bits=0.5,
      verbose=True,
      change_tau=True,
      use_custom_entropy_loss=False,
      use_fast_seq_entropy_loss=False)

optimized_inputs = samples.detach().cpu()
argmax_optimized_input = np.argmax(optimized_inputs, 1)
nuc_seqs = argmax_to_nucleotide(argmax_optimized_input)
visualize_and_save(individual_losses_cache, total_losses, hamming_distances, results_path=results_path, show=True)

# Save nuc_seqs as fasta
fasta_path = os.path.join(results_path, f"no_change.fasta")
with open(fasta_path, "w") as fasta_file:
    for i, seq in enumerate(nuc_seqs):
        fasta_file.write(f">Sequence_{i+1}\n")
        fasta_file.write(f"{seq}\n")

# Save nuc_seqs as txt
txt_path = os.path.join(results_path, f"no_change.txt")
with open(txt_path, "w") as txt_file:
    for i, seq in enumerate(nuc_seqs):
        txt_file.write(f"Sequence {i+1}: {seq}\n")