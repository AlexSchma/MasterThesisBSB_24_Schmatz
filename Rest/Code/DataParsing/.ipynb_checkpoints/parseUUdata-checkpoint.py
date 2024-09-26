# Parse UU data. Return median and average log2Fold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# For further issues, this handles the raw counts from UU data. This
# will be used to predict mean, median ... or for further developments
# in the dynamical model stuff TODO


def load_data():
    data_dict = {"A": [], "B": [], "C": [], "D": [], "G": [], "H": []}

    data_path = "Data/RAW/DEGsUU/time_series_data/GLM_MCR_input.txt"
    store_path = "Data/RAW/DEGsUU/time_series_data/Interim"
    # WTF is that index man xD
    time_in_hours = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
    time_in_index = [1, 2, 3, 5, 7, 9, 13, 15, 17, 19, 21, 23, 26, 28, 30]
    mapping_dict_index = {i: t for t, i in zip(time_in_hours, time_in_index)}

    original_copy = pd.read_csv(data_path, sep="\t").drop(columns="Unnamed: 320")
    df = pd.read_csv(data_path, sep="\t").drop(columns="Unnamed: 320")
    df = df.rename(columns={"Unnamed: 0": "Genes"})
    t0_re = re.compile(r"[A-Z]1\.")

    ### Get the common initial condition
    for key in data_dict:
        for i in range(1, 6):
            # Add the initial condition if it is not already present! (is a copy!)
            if f"{key}1.{i}" not in df.columns:
                df[f"{key}1.{i}"] = df[f"{key}1.{1}"]

    # Put data in long format:
    melted_df = pd.melt(df, id_vars=["Genes"])

    # Split Sample information
    melted_df[["Treatment_time_step", "replicate"]] = melted_df["variable"].str.split(
        ".", expand=True
    )

    melted_df.drop(columns="variable", inplace=True)

    melted_df[["Treatment", "time_step", "_"]] = melted_df[
        "Treatment_time_step"
    ].str.split("(\\d+)", expand=True, regex=True)

    melted_df.drop(columns=["Treatment_time_step", "_"], inplace=True)

    ## Get treatment subgroups:
    for key in data_dict:
        ### Get replicate
        treatment_group = melted_df.loc[melted_df["Treatment"] == key, :].copy()

        for replicate in set(treatment_group["replicate"]):
            replicate_group = treatment_group.loc[
                treatment_group["replicate"] == replicate, :
            ].copy()

            replicate_group.drop(columns=["replicate", "Treatment"], inplace=True)

            replicate_group = replicate_group.pivot(
                index="Genes", columns="time_step", values="value"
            )

            # Order time steps
            replicate_group.columns = replicate_group.columns.astype(int)
            replicate_group = replicate_group.iloc[
                :, np.argsort(replicate_group.columns)
            ]

            # Rename
            replicate_group.columns = [
                mapping_dict_index[i] for i in replicate_group.columns
            ]

            if len(replicate_group.columns) > 10:
                # Make an assert to be sure the handling is correct
                for i in time_in_index:
                    if i == 1:
                        continue
                    columnn_original = f"{key}{i}.{replicate}"
                    h = time_in_hours[[x == i for x in time_in_index].index(True)]

                    if not columnn_original in original_copy.columns:
                        assert not h in replicate_group.columns
                    else:
                        assert np.isclose(
                            replicate_group[h].to_numpy(),
                            original_copy[columnn_original].to_numpy(),
                        ).all(), "Something went wrong"

                replicate_group.to_csv(f"{store_path}/{key}_{replicate}.csv")
                data_dict[key].append(replicate_group.to_numpy())


def get_log2_fold():
    """
    Get log2 fold change for each gene in each treatment
    """

    treatment_name_mapping = {
        "G": "ABA",
        "B": "JA",
        "C": "SA",
        "H": "ABA.JA",
        "D": "SA.JA",
    }

    store_path = "Data/RAW/DEGsUU/time_series_data/Interim"
    lst = os.listdir(store_path)
    lst.sort()
    data_frames_list = []
    for file in lst:  # Get the data ready in a list
        if "A" in file:
            df = pd.read_csv(f"{store_path}/{file}", index_col=0)
        else:
            continue
        data_frames_list.append(df)

    # Concatenate
    mock = pd.concat(data_frames_list, axis=1)
    # Take log 2 per gene
    mock = np.log2(mock + 1)
    # Take mean
    mock = mock.mean(axis=1)

    log2_fold_dict = {}
    median_expression = {}
    # Now for each treatment
    for treat in treatment_name_mapping:
        data_frames_list = []
        for file in lst:
            if treat in file:
                df = pd.read_csv(f"{store_path}/{file}", index_col=0)
            else:
                continue
            data_frames_list.append(df)
        # Concatenate
        with_treat = pd.concat(data_frames_list, axis=1)
        # Take log 2 per gene
        with_treat_log = np.log2(with_treat + 1)
        # Take mean
        with_treat_log = with_treat.mean(axis=1)

        assert (with_treat.index == mock.index).all()
        # asseert the columns are the same
        # Get log2 fold
        log2_fold = with_treat_log - mock
        log2_fold_dict[treatment_name_mapping[treat]] = log2_fold
        median_expression[treatment_name_mapping[treat]] = with_treat.mean(axis=1)
        # print(median_expression[treatment_name_mapping[treat]])
    # Transform to dataframe
    log2_fold_df = pd.DataFrame(log2_fold_dict)
    # Change the index colname to Gene
    log2_fold_df.index.name = "Gene"
    # Save
    log2_fold_df.to_csv("Data/Processed/Y/log2_fold.csv")
    # Save median expression
    median_expression_df = pd.DataFrame(median_expression)
    median_expression_df.index.name = "Gene"
    median_expression_df.to_csv("Data/Processed/Y/median_expression.csv")
    # Save gene names
    gene_names = pd.DataFrame(log2_fold_df.index)
    gene_names.to_csv(
        "Data/Processed/Y/gene_names_log2_fold.csv", index=False, header=False
    )
