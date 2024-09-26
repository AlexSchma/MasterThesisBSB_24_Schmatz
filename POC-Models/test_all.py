from test_cond import *
from test_uncond import *
n_models=10


motifs6mer = ["TTCCCT", "CTACCT", "TGCAGT"]
motifs8mer = ["TTCCCTCG", "CTACCTCC", "TGCAGTGC"]
for _6mer in [True,]:
    if _6mer:
        print("Testing 6mer Motif Models")
        motifs = motifs6mer
    else:
        print("Testing 8mer Motif Models")
        motifs = motifs8mer
    print("Analyzing conditional models")
    run_model_analysis(_6mer=_6mer,motifs=motifs,Base_model_name="BigFilterMaxPoolCond",n_models=n_models,device='cuda')
    print("Analyzing conditional models with embeddings")
    run_model_analysis(_6mer=_6mer,motifs=motifs,Base_model_name="BigFilterMaxPoolCondEmbedding",n_models=n_models,device='cuda')
    print("Analyzing conditional models with embeddings multiplied")
    run_model_analysis(_6mer=_6mer,motifs=motifs,Base_model_name="BigFilterMaxPoolCondEmbeddingMultiply",n_models=n_models,device='cuda')
    print("Analyzing unconditional models")
    run_model_analysis_uncond(_6mer=_6mer,motifs=motifs,Base_model_name="BigFilterMaxPool",n_models=n_models,device='cuda')
    