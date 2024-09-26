import pandas as pd

def compare_csv(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if df1.equals(df2):
        print("The files are identical.")
    else:
        print("The files are different.")

# Example usage
file1 = r"Desktop/Amsterdam/internships/van Dijk/Github_backup/Server_dataset_families_3000_1000_0_1000/gene_families.csv"
file2 = "OneDrive/Dokumente/GitHub/DL-GRN/Data/Processed/gene_families.csv"
compare_csv(file1, file2)


