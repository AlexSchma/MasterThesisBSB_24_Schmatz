import pandas as pd

# Load the data from the text file
file_path = 'Data/RAW/DEGsUU/DEGs_up_down_hormones.txt'
df = pd.read_csv(file_path, sep='\t')

# Initialize the new dataframe with the index and the 5 hormone columns
hormones = ['SA', 'ABA', 'JA', 'SAJA', 'ABAJA']
new_df = pd.DataFrame(index=df.iloc[:, 0], columns=hormones)

# Set the index of the new DataFrame to match the original DataFrame's index
new_df.index = df.iloc[:, 0]

# Create boolean masks for up and down columns
for hormone in hormones:
    up_col = f'{hormone}_up'
    down_col = f'{hormone}_down'
    
    up_mask = df[up_col] == 1
    down_mask = df[down_col] == 1
    
    new_df.loc[up_mask.values, hormone] = 1
    new_df.loc[down_mask.values, hormone] = 0

# Replace empty strings with NaN for clarity
new_df.replace("", pd.NA, inplace=True)

# Rename the columns
new_column_names = {
    'SA': 'SA',
    'ABA': 'ABA',
    'JA': 'MeJA',
    'SAJA': 'SA + MeJA',
    'ABAJA': 'ABA + MeJA'
}
new_df.rename(columns=new_column_names, inplace=True)

# Print the transformed dataframe
print(new_df)

# Save the new dataframe to a CSV file
new_df.to_csv('Data/Processed/over_under_expressed.csv', index_label='Gene')
