import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the counts data
file_path = 'Data/RAW/raw_count_matrix/raw_count_matrix2.txt'
counts_df = pd.read_csv(file_path, delimiter='\t')
counts_df = counts_df.set_index('Gene')

# Replace hyphens with dots in the column names of the counts DataFrame
counts_df.columns = counts_df.columns.str.replace("-", ".")

# Transpose the counts DataFrame to align conditions as row indices
counts_df_t = counts_df.transpose()

# Load the metadata
file_path = 'Data/RAW/metadata.txt'
metadata_df = pd.read_csv(file_path, delimiter='\t')
metadata_df.columns = ['Sample', 'Treatment', 'Time']
metadata_df = metadata_df.set_index('Sample')

# Merge the transposed counts DataFrame with the metadata DataFrame based on the condition identifiers
merged_df = counts_df_t.merge(metadata_df, left_index=True, right_index=True)

# Find the maximum count for each gene across all time points and samples per condition
max_counts_per_condition = merged_df.groupby('Treatment').max()

# Transpose back to the original format
result_df = max_counts_per_condition.transpose()

# Log10 transform the values (adding 1 to each count to handle zeros)
log10_transformed_df = np.log10(result_df + 1)

# Save the transformed data to a new CSV file if needed
log10_transformed_df.to_csv('Data/Processed/log10_transformed_max_counts_per_condition.csv')

# Create a new DataFrame to store gene names and count levels
count_levels = pd.DataFrame(index=log10_transformed_df.index)

# Determine the quartiles for each condition and label the genes
for condition in log10_transformed_df.columns:
    data = log10_transformed_df[condition]
    
    # Calculate quartiles
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    # Label the genes based on quartile ranges
    count_levels[condition] = np.where(data <= q1, 0, 
                                       np.where(data >= q3, 1, ''))

# Combine the count levels into a single column
#count_levels['count_level'] = count_levels.apply(lambda row: '0' if 'low' in row.values else ('0' if 'high' in row.values else ''), axis=1)

# Filter out genes that are in the middle quartile (empty count_level)
#final_count_levels = count_levels[count_levels['count_level'] != '']

# Keep only the gene names and the final count_level
#final_count_levels = final_count_levels[['count_level']]

# Save the result to a new CSV file
count_levels.to_csv('Data/Processed/gene_count_levels.csv')

# Display the number of genes left in the dataset
num_genes_left = len(final_count_levels)
print(f"Number of genes left in the dataset: {num_genes_left}")
print(f"Percentage of genes left: {num_genes_left / len(count_levels) * 100:.2f}%")

# Display the first few rows of the result DataFrame
print(final_count_levels.head())

for condition in log10_transformed_df.columns:
    data = log10_transformed_df[condition]
    
    # Calculate quartiles
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    # Define colors based on quartiles
    bins = np.histogram_bin_edges(data, bins='auto')
    hist, bin_edges = np.histogram(data, bins=bins)
    
    colors = []
    for i in range(len(hist)):
        if bin_edges[i] <= q1:
            colors.append('yellow')  # First quartile
        elif bin_edges[i] > q3:
            colors.append('red')     # Last quartile
        else:
            colors.append('blue')    # Middle 50%

    # Create the histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=bins, edgecolor='black')
    
    # Color the bins
    for i, patch in enumerate(patches):
        plt.setp(patch, 'facecolor', colors[i])
    
    plt.title(f'Log10 Transformed Max Counts for Condition: {condition}')
    plt.xlabel('Log10(Max Count + 1)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'Data/Processed/log10_transformed_max_counts_{condition}.png')
    #plt.show()
    #remember to split between conditions
