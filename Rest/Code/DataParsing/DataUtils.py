#### Functionality to load processed data and split it into training and testing sets
#### with family aware splitting.


from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def family_wise_train_test_splitting(
    gene_families: np.array, test_size=0.8, random_state=42
):
    """
    Split the data into training and testing sets, ensuring that the gene families are not split between the two sets.

    gene_families: The gene families
    test_size: The proportion of the data to be used as the test set
    random_state: The random state for the shuffle split
    ---

    returns: Training and test indices

    """

    X = np.zeros(shape=(len(gene_families), 1))
    Y = np.zeros(shape=(len(gene_families), 1))

    # Create the split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Split the data
    train_index, test_index = next(gss.split(X, Y, groups=gene_families))

    # assert no family is present in both sets
    assert (
        len(
            set(gene_families[train_index]).intersection(set(gene_families[test_index]))
        )
        == 0
    )

    # Return the split data indices
    return train_index, test_index


def process_data(X_names_url: str, Y_url: str, gene_families_url: str):
    """
    Given the raw URLs, process the data and returns the data aligned by gene id (on the row).
    Returns the index (rows) of X, Y and gene_families that are aligned.

    X_names_url: URL of the gene names
    Y_url: URL of the Y data. It must contain a column "Gene" with the gene names.
    ---
    returns: X_indices, Y, gene_families
    """
    # Load the data
    X_names = np.load(X_names_url)

    Y = pd.read_table(Y_url, header=0)

    Y_names = list(Y["Gene"])
    gene_families = pd.read_csv(gene_families_url)
    gene_families_names = list(gene_families["gene_id"])

    intersection = set(X_names).intersection(set(Y_names))
    # assert gene familie names is a superset of the intersection of X and Y,
    # print the difference if it is not

    assert set(gene_families_names).issuperset(intersection), (
        f"there are {len(set(intersection).difference(gene_families_names))} genes in"
        + "the intersection that are not in the gene families, this is likely due to not extending the gene_families with genes outside "
    )

    X_indices = []
    Y_indices = []
    gene_family_indices = []
    for i in range(len(X_names)):
        if X_names[i] in intersection:
            X_indices.append(i)
            Y_indices.append(Y_names.index(X_names[i]))
            gene_family_indices.append(gene_families_names.index(X_names[i]))
    Y = Y.iloc[Y_indices]
    gene_families = gene_families.iloc[gene_family_indices]

    # Now assert that the gene ids are the same, this is, the rows correspond to the same genes
    assert np.all(X_names[X_indices] == gene_families["gene_id"])
    assert np.all(X_names[X_indices] == Y["Gene"])

    # Note that all the data for X (all the embeddings must obey the order given by gene_names)
    return X_indices, Y, gene_families


def kfold_family_wise_splitting(
    X: np.array, Y: np.array, gene_families: np.array, k: int = 5
):
    """
    Split the data into k folds, ensuring that the gene families are not split between the two sets.

    X: The input data
    Y: The labels
    gene_families: The gene families
    k: The number of folds
    ---
    returns: iterable with train and test indices

    """
    # assert gene_id and Y are 1 dimensional (1 column)
    assert len(X.shape) == 2
    assert len(Y.shape) == 1
    assert len(gene_families.shape) == 1

    # Assert that the dimension 0 are the same
    assert X.shape[0] == Y.shape[0] == gene_families.shape[0]

    gss = GroupKFold(n_splits=k)

    return gss.split(
        X, Y, groups=gene_families
    )  # Which is an interable with train and test indices


def make_long_format_data(
    promoter_representation: np.array,
    gene_names: np.array,
    DEG_df: pd.DataFrame,
    at_hormone_level=True,
    only_index=False,
):
    """
    Takes embeddings and treatment **inputs**. Returns a dictionary
    of promoter_treatment with tuples containung the promoter representation
    and a dummy variable. This can be at the treatment level or at the hormone
    level, by default hormone level.

    promoter_representation:
        Some representation of the promoter, the order must be the same as in gene_names. For
        instance one-hot encoded!

    gene_names:
        The gene names of the representation.

    DEG_df:
        A pandas dataframe with the genes and the DEG value per treatment.

    returns:
        A dictionary with ((X_p, X_h), Y), being X1 the promoter representation
        and X_h the dummy for the hormone
    """

    long_format_df = pd.melt(DEG_df, ignore_index=False)
    if at_hormone_level:
        long_format_df["ABA"] = long_format_df["variable"].apply(
            lambda x: int("ABA" in x)
        )
        long_format_df["JA"] = long_format_df["variable"].apply(
            lambda x: int("JA" in x)
        )
        long_format_df["SA"] = long_format_df["variable"].apply(
            lambda x: int("SA" in x)
        )
        treatment_one_hot = long_format_df[["ABA", "SA", "JA"]].to_numpy()
        # IMPORTANT HERE IS THE ORDER!! TODO
    else:
        dummy_variables = pd.get_dummies(long_format_df["variable"])
        treatment_one_hot = dummy_variables.to_numpy().astype(int)

    Y = long_format_df["value"].to_numpy()

    out_dict = {}
    gene_names_long = []
    for j, i in tqdm(enumerate(long_format_df.index)):
        index = np.where(gene_names == i)[0]
        # if index is not empty
        if len(index) > 0:
            if only_index:
                out_dict[j] = ((index, treatment_one_hot[j]), Y[j])
            else:
                out_dict[j] = (
                    (promoter_representation[index], treatment_one_hot[j]),
                    Y[j],
                )
            gene_names_long.append(i)
    print("Number of 'observations' in the dictionary:", len(out_dict))
    return out_dict, gene_names_long


if __name__ == "__main__":
    # TODO make a proper function here
    # Load the data
    representation = np.load("Data/Processed/One-Hot/One_hot_encoded.npy")
    X_names = np.load("Data/Processed/gene_names.npy")
    deg_table = pd.read_table(
        "Data/RAW/DEGsUU/DEGs_treatments_vs_mock.txt", index_col=0
    )
    xy_dict, gene_names_long = make_long_format_data(
        representation, X_names, deg_table, at_hormone_level=True
    )

    with open("Data/Processed/XY_formats/One_hot_encoded.pkl", "wb") as f:
        pickle.dump(xy_dict, f)

    np.save(
        "Data/Processed/XY_formats/gene_names_one_hot.npy", np.array(gene_names_long)
    )

    representation = np.load(
        "Data/Processed/Embeddings/embeddings_agro_nt_full_2024-04-06_19-39-14.npy"
    ).astype(np.float16)
    X_names = np.load("Data/Processed/gene_names.npy")
    deg_table = pd.read_table(
        "Data/RAW/DEGsUU/DEGs_treatments_vs_mock.txt", index_col=0
    )
    xy_dict, gene_names_long = make_long_format_data(
        representation, X_names, deg_table, at_hormone_level=True, only_index=True
    )

    with open("Data/Processed/XY_formats/AgroNT.pkl", "wb") as f:
        pickle.dump(xy_dict, f)

    np.save(
        "Data/Processed/XY_formats/gene_names_AgroNT.npy", np.array(gene_names_long)
    )
    exit()

    X_index, Y, gene_family = process_data(
        "Data/Processed/gene_names.npy",
        "Data/RAW/DEGsUU/time_series_data/DEGs_treatments_vs_mock.txt",
        "Data/Processed/gene_families.csv",
    )

    Y = Y.iloc[:, 1].to_numpy()
    gene_families = gene_family["family_id"].to_numpy()
    X = X[X_index]

    # Alright. SO we have X, Y and gene_families
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        gene_families_train,
        gene_families_test,
    ) = family_wise_train_test_splitting(X, Y, gene_families)

    (
        X_train,
        X_val,
        Y_train,
        Y_val,
        gene_families_train,
        gene_families_val,
    ) = family_wise_train_test_splitting(X_train, Y_train, gene_families_train)

    for train_index, test_index in kfold_family_wise_splitting(
        X_train, Y_train, gene_families_train
    ):
        # assert no family is present in both sets
        assert (
            len(
                set(gene_families_train[train_index]).intersection(
                    set(gene_families_train[test_index])
                )
            )
            == 0
        )
