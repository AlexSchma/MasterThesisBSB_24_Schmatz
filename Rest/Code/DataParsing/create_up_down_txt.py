import pandas as pd

# Specify the file path
txt_path = 'DEGs_up_down_hormones.txt'
data_path = 'Data/RAW/DEGsUU'
file_path = f'{data_path}/{txt_path}'
# Load the file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')
# Rename the first column as "Gene"
df.columns = ["Gene"] + list(df.columns[1:])

# Define hormones
hormones = ["SA", "ABA", "JA", "SAJA", "ABAJA"]

# Create a dictionary to hold the new dataframes
hormone_dfs = {}
outcome_dfs = []
genes = 0
for hormone in hormones:
    # Filter relevant columns for the hormone
    relevant_cols = ["Gene", f"{hormone}_up", f"{hormone}_down"]
    hormone_df = df[relevant_cols].copy()
    
    # Apply the transformation logic
    hormone_df[f"{hormone}_up_down"] = hormone_df.apply(
        lambda row: 1 if row[f"{hormone}_up"] == 1 else (0 if row[f"{hormone}_down"] == 1 else None), axis=1
    )
    
    # Drop the rows where both up and down are 0
    hormone_df = hormone_df.dropna(subset=[f"{hormone}_up_down"])
    
    # Drop the original up and down columns
    hormone_df = hormone_df.drop(columns=[f"{hormone}_up", f"{hormone}_down"])
    
    # Save to the dictionary
    hormone_dfs[hormone] = hormone_df

    # Print the number of genes per hormone
    print(f"{hormone}: {len(hormone_df)} genes")
    genes += len(hormone_df)
    outcome_dfs.append(hormone_df)
print(f"Total: {genes} genes")
outcome_df = pd.concat(outcome_dfs, axis=1, join='outer')
outcome_df.to_csv(f"{data_path}/outcome_df.txt", index=False)
# Save the dataframes to CSV files
for hormone, df in hormone_dfs.items():
    df.to_csv(f"{data_path}/{hormone}_response_up_down.txt", index=False)
