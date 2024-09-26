import pandas as pd

# Specify the file path
txt_path = 'DEGs_up_down_hormones.txt'
data_path = 'Data/RAW/DEGsUU'
file_path = f'{data_path}/{txt_path}'
# Load the file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')

# Define the hormones
hormones = ["SA", "JA", "ABA", "SAJA", "ABAJA"]

# Function to create a new dataframe for a specific hormone
def create_hormone_df(df, hormone):
    gene_name_col = df.columns[0]
    up_col = f"{hormone}_up"
    down_col = f"{hormone}_down"
    
    new_df = df[[gene_name_col, up_col, down_col]].copy()
    new_df['response'] = 1  # Default response
    
    # Update response based on conditions
    new_df.loc[new_df[up_col] == 1, 'response'] = 2
    new_df.loc[new_df[down_col] == 1, 'response'] = 0
    
    # Select only gene name and response columns
    new_df = new_df[[gene_name_col, 'response']]
    new_df.columns = ['Gene', 'Response']
    return new_df

# Create a dictionary to hold the dataframes
hormone_dfs = {hormone: create_hormone_df(df, hormone) for hormone in hormones}

# Save each dataframe as a txt file
for hormone, hormone_df in hormone_dfs.items():
    print(hormone_df['Response'].value_counts())
    hormone_df.to_csv(f'{data_path}/{hormone}_response.txt', sep='\t', index=False)