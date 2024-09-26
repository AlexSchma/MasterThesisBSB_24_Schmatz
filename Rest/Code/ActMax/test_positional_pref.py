import torch
import numpy as np
from Analysis_Utils import *
from Data_Utils import *
from FunProseModels import *
import os
import json
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tqdm as tqdm

def random_dna_sequence(seq_len, num_seqs):
    return [''.join(np.random.choice(['A', 'C', 'G', 'T'], size=seq_len)) for i in range(num_seqs)]

def get_motif_insertion(motif_insertions, inter_motif_dist, motif):
    insterted_seq = ""
    for i in range(motif_insertions):
        insterted_seq += motif
        if i < motif_insertions-1:
            insterted_seq += ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=inter_motif_dist))
    return insterted_seq

def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'T': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([mapping[base] for base in seq])

def create_pos_bias_seqs(seq_len, num_seqs, pos, motif_insertions, inter_motif_dist, motif):
    len_motif_insertion =(len(motif)+inter_motif_dist)*motif_insertions-inter_motif_dist
    end_position_of_insertion = len_motif_insertion + pos
    assert end_position_of_insertion <= seq_len, f"Motifs exceed sequence length, end position of insertion is {end_position_of_insertion} and sequence length is {seq_len}"
    random_seqs = random_dna_sequence(seq_len, num_seqs)
    inserted_seqs = []
    for seq in random_seqs:
        seq = seq[:pos] + get_motif_insertion(motif_insertions, inter_motif_dist, motif) + seq[pos+len_motif_insertion:]
        inserted_seqs.append(one_hot_encode(seq))
    return np.array(inserted_seqs).transpose(0, 2, 1)

def load_model(state_dict_path, device, seq_length):
    model = get_funprose_multi_class(seq_length=seq_length)
    model_state_dict = torch.load(state_dict_path)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model

def get_test_loader(seq_len, num_seqs,grid_size, motif_insertions, inter_motif_dist, motif):
    test_seqs = []
    len_motif_insertion =(len(motif)+inter_motif_dist)*motif_insertions-inter_motif_dist
    last_insert_pos = seq_len - len_motif_insertion
    test_seqs = None
    for i in range(0,last_insert_pos+1,grid_size):
        if test_seqs is None:
            test_seqs = create_pos_bias_seqs(seq_len=seq_len, num_seqs=num_seqs, pos=i, motif_insertions=motif_insertions, inter_motif_dist=inter_motif_dist, motif=motif)
        else:
            test_seqs = np.concatenate((test_seqs, create_pos_bias_seqs(seq_len=seq_len, num_seqs=num_seqs, pos=i, motif_insertions=motif_insertions, inter_motif_dist=inter_motif_dist, motif=motif)), axis=0)
    test_seqs = torch.tensor(test_seqs).float()
    test_dataset = TensorDataset(test_seqs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_seqs, shuffle=False)
    return test_loader

def run_test_loop(test_loader, model, device, cond_embedding):
    all_preds = []
    for batch in tqdm.tqdm(test_loader):
        seqs = batch[0].to(device)
        num_seqs = seqs.shape[0]
        cond_tensor = torch.tensor(cond_embedding).repeat(num_seqs, 1).float().to(device)
        preds = model(seqs, cond_tensor)
        preds = F.softmax(preds, dim=1)
        preds = preds.cpu().detach().numpy()
        all_preds.append(preds)
    return np.array(all_preds)

def motif_loop(seq_len, num_seqs, grid_size, motif_insertions, inter_motif_dist, model_state_dict_path, cond_embedding_dict,motif_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_model(model_state_dict_path, device, seq_len)
    print(f"model loaded")
    pred_dict = {}
    for key in cond_embedding_dict.keys():
        cond_tensor = cond_embedding_dict[key]
        motif = motif_dict[key]
        print(f"Generating Sequences for {key} with motif: {motif}")
        test_loader = get_test_loader(seq_len, num_seqs, grid_size, motif_insertions, inter_motif_dist, motif)
        print(f"Running test loop for {key} with motif: {motif}")
        pred_dict[key]=run_test_loop(test_loader, model, device, cond_tensor)
    return pred_dict

def plot_means_with_std(pred_dict, save_path=None):
    plt.figure(figsize=(10, 6))

    for key, value in pred_dict.items():
        mean_values = np.mean(value[:, :, 2], axis=1)
        std_values = np.std(value[:, :, 2], axis=1)
        
        x = np.arange(len(mean_values))
        plt.plot(x, mean_values, label=key)
        plt.fill_between(x, mean_values - std_values, mean_values + std_values, alpha=0.2)

    plt.xlabel('Relative Position of Motif Insertion')
    plt.ylabel('Mean Values with Std')
    plt.title('Mean and Std of Conditions')
    plt.legend()
    plt.grid(True)
    if save_path:
        save_path = os.path.join(save_path, "mean_std_plot.png")
        plt.savefig(save_path)
    plt.show()

def save_pred_dict(pred_dict, save_path):
    save_path = os.path.join(save_path, "pred_dict.npz")
    np.savez(save_path, **pred_dict)

cond_embedding_dict = {
        "SA": [0, 1, 0],
        "ABA": [1, 0, 0],
        "JA": [0, 0, 1],   
    }
motif_dict ={"SA":'ACGTTGCA',"JA":'TGCATGCA',"ABA":'GTTGCAAC'}
#print(create_pos_bias_seqs(seq_len=5,num_seqs= 2,pos= 2, motif_insertions=1, inter_motif_dist= 0,motif= 'TT').shape)
                                     
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_loader = get_test_loader(seq_len=8, num_seqs=3, grid_size=1, motif_insertions=2, inter_motif_dist=1, motif='AA')
# print(len(test_loader))
# for batch in test_loader:
#     print("batch[0}.shape: ",batch[0].shape)
#     print("batch[0]: ",batch[0])
#     print("batch len: ",len(batch))
#     print(batch)
#     print("_"*50)
save_path = "test_seqs/test_positional"
if not os.path.exists(save_path):
    os.makedirs(save_path)
pred_dict = motif_loop(seq_len=1000, num_seqs=100, grid_size=1, motif_insertions=3, inter_motif_dist=5, model_state_dict_path='models/sim3_1000_model_best_model.pth', cond_embedding_dict=cond_embedding_dict, motif_dict=motif_dict)
save_pred_dict(pred_dict, save_path)

plot_means_with_std(pred_dict,save_path=save_path)
