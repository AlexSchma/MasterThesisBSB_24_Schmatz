from train_cond_models import train_and_save as train_cond
from train_uncond_models import train_and_save as train_uncond
from train_cond_embedding import train_and_save as train_cond_emb
from train_cond_embedding_multiply import train_and_save as train_cond_emb_mult
from Data_Utils import create_synth_data, create_synth_data_uncond
print("imported all modules and functions")
n_models = 10
epochs = 50
print("Training 6mer Motif Models")
_6mer = True

motifs = ["TTCCCT", "CTACCT", "TGCAGT"]
print("Creating synthetic data for 6mer models")
train_loader_uncond, val_loader_uncond, test_loader_uncond = create_synth_data_uncond(motifs=motifs)
train_loader, val_loader, test_loader = create_synth_data(motifs=motifs)
# print("Training unconditional models")
# train_uncond(train_loader_uncond, val_loader_uncond,epochs=epochs,n_models=n_models,_6mer=_6mer)
# print("Training conditional models")
# train_cond(train_loader, val_loader,epochs=epochs,n_models=n_models,_6mer=_6mer)
# print("Training models with conditional embeddings")
# train_cond_emb(train_loader, val_loader,epochs=epochs,n_models=n_models,_6mer=_6mer)
print("Training models with conditional embeddings multiplied")
train_cond_emb_mult(train_loader, val_loader,epochs=epochs,n_models=n_models,_6mer=_6mer)

print("Training 8mer Motif Models")
_6mer = False

motifs = ["TTCCCTCG", "CTACCTCC", "TGCAGTGC"]
print("Creating synthetic data for 8mer models")
train_loader_uncond, val_loader_uncond, test_loader_uncond = create_synth_data_uncond(motifs=motifs)
train_loader, val_loader, test_loader = create_synth_data(motifs=motifs)
print("Training unconditional models")
train_uncond(train_loader_uncond, val_loader_uncond,epochs=epochs,n_models=n_models,_6mer=_6mer)
print("Training conditional models")
train_cond(train_loader, val_loader,epochs=epochs,n_models=n_models,_6mer=_6mer)
print("Training models with conditional embeddings")
train_cond_emb(train_loader, val_loader,epochs=epochs,n_models=n_models,_6mer=_6mer)
print("Training models with conditional embeddings multiplied")
train_cond_emb_mult(train_loader, val_loader,epochs=epochs,n_models=n_models,_6mer=_6mer)