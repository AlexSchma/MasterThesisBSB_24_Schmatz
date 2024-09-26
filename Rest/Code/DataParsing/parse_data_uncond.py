"""
    Get the arguments for the dataset to be constructed.
    Generates a folder with X_train, X_test, X_val, y_train, y_test, y_val.
    Y should be a .csv with gene_name, family_id, outcome.
    X is either a .csv with gene_name, famili_id, "<gene_name>.npy" or a .npy which is the
    concatenation of all the <gene_name>.npy files.
"""

import sys

sys.path.append(".")
from Code.DataParsing.geneFamilies import generate_gene_families, add_all_genes
from Code.Representation.generate_representations import (
    load_promoters,
    generate_one_hot_representation,
    save_gene_names,
    load_and_concatenate_introns,
    generate_kmer_one_hot_vectors,
)
import argparse
import os
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm


def parse_bool(s: str) -> bool:
    try:
        return {"true": True, "false": False}[s.lower()]
    except KeyError as exc:
        raise ValueError("Invalid boolean value") from exc


def parse_TFs(data_matrix: pd.DataFrame, TFs="Data/RAW/TF_mapping.txt"):
    """
    Given the data matrix, returns a dictionary with column name as a key and an address to a .npy
    array containing the expression of the selected transcription factor as a value.
    """
    TF_mapping = pd.read_table(TFs)["TF"]
    # Get unique TFs
    TFs = list(set(TF_mapping))

    # construct a dictionary with the column name as keys and the addresses to the .npy files as values
    # get TF coordinates in the data matrix
    tf_index = [data_matrix.index.get_loc(tf) for tf in TFs]

    assert all(
        data_matrix.iloc[tf_index].index.values == TFs
    ), "something went wrong with parsing TFs"

    TF_dict = {}
    data_matrix.columns = [i.replace(".", "_") for i in data_matrix.columns]

    for treatment in data_matrix.columns:
        TF_activation = data_matrix[treatment].iloc[tf_index].to_numpy()[np.newaxis, :]
        # save the TF activation as a .npy file
        np.save(f"Data/Processed/TreatmentInput/TFs/{treatment}.npy", TF_activation)
        TF_dict[treatment] = f"Data/Processed/TreatmentInput/TFs/{treatment}.npy"

    return TF_dict


def generate_treatment_representations(directory, matrix_form, TFs=None):
    """
    Generate the representations of the treatments.

    Parameters
    ----------
    directory : str
        The directory where the representations will be stored.
    TFs : list (optional) default=None
        The list of TFs differentially expressed in the treatment.
        (Essentially this should be a dictionary with the TFs as keys and the
        if they are differential expressed in the treatment as values.)
    """
    os.makedirs(directory, exist_ok=True)
    if TFs is not None:
        raise NotImplementedError("This function is not implemented yet.")
    if matrix_form:
        treatment = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]])
        np.save(os.path.join(directory, "matrix.npy"), treatment)
    if TFs is None:
        # MeJA/SA/ABA
        # [0, 0, 0] -> None
        # [1, 0, 0] -> MeJA
        # [0, 1, 0] -> SA
        # [0, 0, 1] -> ABA
        treatment_mapping = {
            # "A": np.array([0, 0, 0]), # Because this one is not really used. It is used to compare.
            "B": np.array([1, 0, 0]),
            "C": np.array([0, 1, 0]),
            "D": np.array([1, 1, 0]),
            "G": np.array([0, 0, 1]),
            "H": np.array([1, 0, 1]),
        }
        for treatment, mapping in treatment_mapping.items():
            np.save(os.path.join(directory, f"{treatment}.npy"), mapping)


def get_treatment_representatio(treatment_name, directory) -> str:
    """
    Returns address of the representation.

    Parameters
    ----------
    treatment_name : str
    Returns
    -------
    The URL of the representation of the treatment, a .npy file.
    """
    return os.path.join(directory, f"{treatment_name}.npy")


def construct_dataset(
    promoter_rep_urls: list,
    treatment_input_url,
    treatments=["G"],
    hormones = ["ABA", "ABAJA","SA","SAJA","JA"],
    outcome_url="Data/RAW/DEGsUU/treatments_response.txt",
    family_url="Data/Processed/gene_families.csv",
    raw_counts_url="Data/RAW/DEGsUU/GLM_MCR_input.txt",
    raw_counts=False,
    matrix_like=False,
    dataset_url="Data/Processed/dataset.csv",
    mapping = {"Response": "G"}
):
    """
    Construct the dataset.

    TODO IMPROVE THIS FUNCTION, IS AWKWARD TO USE.
    """
    dataset_dict = {}
    for hormone in hormones:
        dataset_dict[hormone] = []
    gene_names = []
    family_df = pd.read_csv(family_url, index_col="gene_id")
    if raw_counts:
        outcome_df = pd.read_table(raw_counts_url, index_col=0)
        outcome_df.dropna(axis=1, inplace=True)
        TF_dict = parse_TFs(outcome_df)
    else:
        print("Using the DEGs from the experiments")
        outcome_df_dict = {}
        for hormone in hormones:
            outcome_df_dict[hormone] = pd.read_csv(outcome_url.replace("treatments", hormone), sep="\t", index_col=0)
            outcome_df_dict[hormone] = outcome_df_dict[hormone].rename(columns=mapping)
    for i in tqdm(promoter_rep_urls):
        if isinstance(i, tuple):
            gene_name = i[0].split(".")[0].split("/")[-1]
        else:
            gene_name = i.split(".")[0].split("/")[-1]
        gene_names.append(gene_name)
        for hormone in hormones:
            if gene_name not in outcome_df_dict[hormone].index:
                continue
        if raw_counts:  # Conditions here is treaatment x timepoints
            for conditions in outcome_df.columns:
                prom_address = i
                y = outcome_df.loc[
                    gene_name, conditions
                ]  # This is the raw count data mRNA
                x = (prom_address, TF_dict[conditions])
                z = family_df.loc[gene_name, "family_id"]
                dataset.append((x, y, z))

        elif matrix_like:
            prom_address = i
            input_address = os.path.join(treatment_input_url, "matrix.npy")
            x = (prom_address, input_address)
            treatment_outcomes = np.zeros((len(treatments)))
            z = family_df.loc[gene_name, "family_id"]

            for treatment in treatments:
                y = outcome_df.loc[gene_name, treatment]
                # The family has to be increased first TODO
                treatment_outcomes[treatments.index(treatment)] = y
            dataset.append((x, list(treatment_outcomes), z))

        else:
            for treatment in treatments:
                prom_address = i
                # This should iterate over all the genes available based on TAIR data (not on experiments)
                # Because, indeed the input is the same for all the genes!
                input_address = get_treatment_representatio(
                    treatment, treatment_input_url
                )
                
                x = (prom_address, input_address)
                # The family has to be increased first TODO
                z = family_df.loc[gene_name, "family_id"]
                for hormone in hormones:
                    y = outcome_df_dict[hormone].loc[gene_name, treatment]
                    dataset_dict[hormone].append((x, y, z))
    for hormone in hormones:
        missing_genes = set(outcome_df_dict[hormone].index) - set(gene_names)
        print(f"Gene names in outcome_df but not in promoter_rep_dir for {hormone}:")
        print(len(missing_genes))
    # Save the dataset as .csv
    for hormone in hormones:
        dataset_df = pd.DataFrame(dataset_dict[hormone], columns=["X", "Y", "Z"])
        dataset_df.to_csv(dataset_url.replace("dataset", f"dataset_{hormone}"), index=False)
        print(f"Dataset constructed and saved as {dataset_url.replace('dataset', f'dataset_{hormone}')}")


# 0. Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--upstreamTSS", type=int, default=3000, help="Number of base pairs upstream of TSS"
)
parser.add_argument(
    "--downstreamTSS",
    type=int,
    default=1000,
    help="Number of base pairs downstream of TSS",
)
parser.add_argument(
    "--upstreamTTS", type=int, default=0, help="Number of base pairs upstream of TTS"
)
parser.add_argument(
    "--downstreamTTS",
    type=int,
    default=1000,
    help="Number of base pairs downstream of TTS",
)
# parse output diretory for the dna sequences
parser.add_argument(
    "--outputDirRawDNA",
    type=str,
    default="Data/RAW",
    help="Output directory for the DNA sequences",
)
parser.add_argument(
    "--outputDirProcessedDNA",
    type=str,
    default="Data/Processed/One_hot_1500_100_100_700",
    help="Output directory for the processed DNA sequences",
)
# Where to store the representations per treamtent
parser.add_argument(
    "--outputDirTreatmentInput",
    type=str,
    default="Data/Processed/TreatmentInput_uncond",
    help="Output directory for the processed DNA sequences",
)
# parse for one-hot encoding
parser.add_argument(
    "--oneHot",
    type=parse_bool,
    default=True,
    help="Whether to one-hot encode the DNA sequences",
)
parser.add_argument(
    "--kmers",
    type=int,
    default=6,
    help="The number of kmers to use if one-hot encoding is false",
)
# parse for concatenated
parser.add_argument(
    "--concatenated",
    type=parse_bool,
    default=False,
    help="Whether to concatenate the DNA sequences",
)
parser.add_argument(
    "--rawCounts",
    type=parse_bool,
    default=False,
    help="Whether to use raw counts for the dataset",
)

parser.add_argument(  # TODO
    "--introns",
    type=parse_bool,
    default=True,
    help="Whether to include introns in the dataset",
)
parser.add_argument(
    "--parallel",
    type=parse_bool,
    default=False,
    help="If treatment embeddings should be provided as a matrix",
)

args = parser.parse_args()
# 1. Generate the gene families file if it does not exist.
if not os.path.exists("Data/Processed/gene_families.csv"):
    user_input = input(
        "Do you want to run BLAST and Markov Clustering to construct gene fam.?"
        + "(yes/no): "
    )
    if user_input.lower() == "yes":
        generate_gene_families()
    else:
        print("Skipping gene families construction.")

# 2. Gather the DNA with the provided arguments.
cmd = [
    "Code/DataParsing/parse_dna.sh",
    str(args.upstreamTSS),
    str(args.downstreamTSS),
    str(args.upstreamTTS),
    str(args.downstreamTTS),
    args.outputDirRawDNA,
]
#subprocess.run(cmd, check=True)
print(f"DNA sequences have been parsed, sequences are in {args.outputDirRawDNA}")
# 3. Optional, one-hot encoding DNA
if args.oneHot:
    # get promoter sequences in python, dict
    # with gene name as key and DNA string as value
    url_promoters_tss = f"{args.outputDirRawDNA}/promoters_{args.upstreamTSS}up_{args.downstreamTSS}down_TSS.fasta"
    url_promoters_tts = f"{args.outputDirRawDNA}/promoters_{args.upstreamTTS}up_{args.downstreamTTS}down_TTS.fasta"
    url_introns = f"{args.outputDirRawDNA}/introns.fasta"

    promoter_dict_tss = load_promoters(url_promoters_tss)
    promoter_dict_tts = load_promoters(url_promoters_tts)
    if args.introns:
        introns_dict = load_and_concatenate_introns(url_introns)
        # import matplotlib.pyplot as plt
        length_introns = [len(x) for x in introns_dict.values()]

        # count number of introns above 2000
        print(
            "Number of introns above 2000: ", len([x for x in length_introns if x > 2000])
        )
    # print("Introns length statistics")
    # print("Mean: ", np.mean(length_introns))
    # print("Median: ", np.median(length_introns))
    # print("Max: ", np.max(length_introns))
    # print("Min: ", np.min(length_introns))
    # print("Std: ", np.std(length_introns))
    ## print percentiles
    # print("Percentiles")
    # print("25th: ", np.percentile(length_introns, 25))
    # print("75th: ", np.percentile(length_introns, 75))
    # print("90th: ", np.percentile(length_introns, 90))
    # print("95th: ", np.percentile(length_introns, 95))
    # print("99th: ", np.percentile(length_introns, 99))
    ##plt.hist([len(x) for x in introns_dict.values()], 1000)
    ##plt.show()
    ### Get the name of the gene with the longest intron
    # max_length = np.argmax(length_introns)
    # print("Gene with the longest intron: ", list(introns_dict.keys())[max_length])
    # exit()
    # exit()
    # Right now this is NOT excluding non-ACGT characters. In the one_hot encoding
    # those characters are just 0s.

    gene_names_promoters = list(promoter_dict_tss.keys())
    add_all_genes(
        gene_families_url="Data/Processed/gene_families.csv",
        gene_names=gene_names_promoters,
    )

    # one-hot encode the DNA sequences
    # Check if folder is empty
    if args.concatenated:
        raise NotImplementedError(
            "Concatenated one-hot encoding is not implemented yet."
        )
        # one_hot_promoters = generate_one_hot_representation(promoter_dict,
        #                                                    concatenated_array=True,
        #                                                    folder=args.outputDirProcessedDNA)
    else:
        if not os.path.exists(args.outputDirProcessedDNA + "/" + "TSS"):
            os.makedirs(args.outputDirProcessedDNA + "/" + "TSS", exist_ok=True)
        if not os.path.exists(args.outputDirProcessedDNA + "/" + "TTS"):
            os.makedirs(args.outputDirProcessedDNA + "/" + "TTS", exist_ok=True)

        one_hot_promoters_tss, _ = generate_one_hot_representation(
            promoter_dict_tss,
            concatenated_array=False,
            folder=args.outputDirProcessedDNA + "/" + "TSS",
        )
        if args.upstreamTTS != 0 or args.downstreamTTS != 0:
            one_hot_promoters_tts, _ = generate_one_hot_representation(
                promoter_dict_tts,
                concatenated_array=False,
                folder=args.outputDirProcessedDNA + "/" + "TTS",
            )

        if args.introns:
            one_hot_introns, max_length_introns = generate_one_hot_representation(
                introns_dict,
                concatenated_array=False,
                folder=args.outputDirProcessedDNA + "/" + "introns",
                maxlength_allowed=2000,
            )
            # create a null intron
            np.save(
                args.outputDirProcessedDNA + "/" + "introns" + "/" + "null_intron.npy",
                np.zeros((max_length_introns, 4)),
            )
        print(
            f"DNA sequences have been one-hot encoded. Stored in {args.outputDirProcessedDNA + '/' + 'TSS'}, {args.outputDirProcessedDNA + '/' + 'TTS'} and {args.outputDirProcessedDNA + '/' + 'introns'}"
        )
        print("Number of one-hot encoded sequences (i.e. different genes): ")
        promoters_tss = os.listdir(args.outputDirProcessedDNA + "/" + "TSS")
        if args.upstreamTTS != 0 or args.downstreamTTS != 0:
            promoters_tts = os.listdir(args.outputDirProcessedDNA + "/" + "TTS")
        if args.introns:
            introns = os.listdir(args.outputDirProcessedDNA + "/" + "introns")
        if args.upstreamTTS != 0 or args.downstreamTTS != 0:
            assert set(promoters_tss) == set(
                promoters_tts
            ), "TSS and TTS promoters are not the same."
        if args.introns:
            print("Number of genes with at least one intron: ", len(introns))

else:
    print("One-hot encoding is not selected.")
    # get promoter sequences in python, dict
    # with gene name as key and DNA string as value
    url_promoters_tss = f"{args.outputDirRawDNA}/promoters_{args.upstreamTSS}up_{args.downstreamTSS}down_TSS.fasta"
    url_promoters_tts = f"{args.outputDirRawDNA}/promoters_{args.upstreamTTS}up_{args.downstreamTTS}down_TTS.fasta"
    url_introns = f"{args.outputDirRawDNA}/introns.fasta"

    promoter_dict_tss = load_promoters(url_promoters_tss)
    promoter_dict_tts = load_promoters(url_promoters_tts)
    introns_dict = load_and_concatenate_introns(url_introns)
    # import matplotlib.pyplot as plt
    length_introns = [len(x) for x in introns_dict.values()]

    # count number of introns above 2000
    print(
        "Number of introns above 2000: ", len([x for x in length_introns if x > 2000])
    )

    gene_names_promoters = list(promoter_dict_tss.keys())
    add_all_genes(
        gene_families_url="Data/Processed/gene_families.csv",
        gene_names=gene_names_promoters,
    )

    if not os.path.exists(args.outputDirProcessedDNA + "/" + "TSS"):
        os.makedirs(args.outputDirProcessedDNA + "/" + "TSS", exist_ok=True)
    if not os.path.exists(args.outputDirProcessedDNA + "/" + "TTS"):
        os.makedirs(args.outputDirProcessedDNA + "/" + "TTS", exist_ok=True)

    one_hot_promoters_tss = generate_kmer_one_hot_vectors(
        promoter_dict_tss,
        args.kmers,
        folder=args.outputDirProcessedDNA + "/" + "TSS",
        reverse_complement=True,
    )
    if args.upstreamTTS != 0 or args.downstreamTTS != 0:
        one_hot_promoters_tts = generate_kmer_one_hot_vectors(
            promoter_dict_tts,
            args.kmers,
            folder=args.outputDirProcessedDNA + "/" + "TTS",
            reverse_complement=True,
        )

    if args.introns:
        one_hot_introns = generate_kmer_one_hot_vectors(
            introns_dict,
            args.kmers,
            folder=args.outputDirProcessedDNA + "/" + "introns",
            reverse_complement=True,
        )
        # create a null intron
        np.save(
            args.outputDirProcessedDNA + "/" + "introns" + "/" + "null_intron.npy",
            np.zeros((max_length_introns, 4)),
        )
    print(
        f"DNA sequences have been {args.kmers}-ers encoded. Stored in {args.outputDirProcessedDNA + '/' + 'TSS'}, {args.outputDirProcessedDNA + '/' + 'TTS'} and {args.outputDirProcessedDNA + '/' + 'introns'}"
    )
    print("Number of one-hot encoded sequences (i.e. different genes): ")
    promoters_tss = os.listdir(args.outputDirProcessedDNA + "/" + "TSS")
    if args.upstreamTTS != 0 or args.downstreamTTS != 0:
        promoters_tts = os.listdir(args.outputDirProcessedDNA + "/" + "TTS")
    if args.introns:
        introns = os.listdir(args.outputDirProcessedDNA + "/" + "introns")
    if args.upstreamTTS != 0 or args.downstreamTTS != 0:
        assert set(promoters_tss) == set(
            promoters_tts
        ), "TSS and TTS promoters are not the same."
    if args.introns:
        print("Number of genes with at least one intron: ", len(introns))

# Now, since the introns are not in all genes, I need to create the tuple properly.
if args.introns:
    tuple_list = []
    for gene in tqdm(one_hot_promoters_tss.keys()):
        if gene not in one_hot_introns:
            one_hot_introns[gene] = os.path.join(
                args.outputDirProcessedDNA, "introns", "null_intron.npy"
            )
        tuple_list.append(
            (
                one_hot_promoters_tss[gene],
                one_hot_promoters_tts[gene],
                one_hot_introns[gene],
            )
        )
else:
    tuple_list = []
    for gene in tqdm(one_hot_promoters_tss.keys()):
        if args.upstreamTTS != 0 or args.downstreamTTS != 0:
            tuple_list.append(
                (one_hot_promoters_tss[gene], one_hot_promoters_tts[gene])
            )
        else:
            tuple_list.append(one_hot_promoters_tss[gene])
# 4. Construct dataset
generate_treatment_representations(
    args.outputDirTreatmentInput, TFs=None, matrix_form=args.parallel
)
# hormone = "SA"
# mapping_hormone = {"Response": "G"}
construct_dataset(
    tuple_list,
    args.outputDirTreatmentInput,
    raw_counts=args.rawCounts,
    matrix_like=args.parallel,

)
