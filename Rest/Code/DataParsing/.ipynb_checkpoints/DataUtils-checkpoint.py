#### Functionality to load processed data and split it into training and testing sets
#### with family aware splitting.


from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import numpy as np
import pandas as pd


def family_wise_train_test_splitting(
    X: np.array,
    Y: np.array,
    gene_families: np.array,
    test_size=0.2,
    random_state=42,
    return_index=False,
):
    """
    Split the data into training and testing sets, ensuring that the gene families are not split between the two sets.

    X: The input data
    Y: The labels
    gene_families: The gene families
    test_size: The proportion of the data to be used as the test set
    random_state: The random state for the shuffle split
    return_index: If True, return the indices of the split instead of the data
    ---
    returns: The training and testing sets:
        X_train, X_test, Y_train, Y_test, gene_families_train, gene_families_test

    """
    # assert gene_id and Y are 1 dimensional (1 column)
    assert len(X.shape) == 2
    assert len(Y.shape) == 1
    assert len(gene_families.shape) == 1

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

    # Return the split data
    if return_index:
        return train_index, test_index
    else:
        return (
            X[train_index],
            X[test_index],
            Y[train_index],
            Y[test_index],
            gene_families[train_index],
            gene_families[test_index],
        )


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

    assert set(gene_families_names).issuperset(
        intersection
    ), f"there are {len(set(intersection).difference(gene_families_names))} genes in" + \
    "the intersection that are not in the gene families, this is likely due to not extending the gene_families with genes outside "

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

    gss = GroupKFold(n_splits=k)

    return gss.split(
        X, Y, groups=gene_families
    )  # Which is an interable with train and test indices


if __name__ == "__main__":
    # Load the data
    X = np.load("Data/Processed/embeddings_agro_nt_average_2024-02-04_11-52-42.npy")

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
