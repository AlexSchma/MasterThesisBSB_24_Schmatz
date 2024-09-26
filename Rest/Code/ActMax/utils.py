import numpy as np
import pandas as pd
import tqdm as tqdm
import torch
import torch.nn as nn

# Function to one-hot encode DNA sequences with padding and handling unknown bases
def onehot(sequences):
    code = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'P': [0, 0, 0, 0],  # Padding
            'N': [0.25, 0.25, 0.25, 0.25]}
    
    encoded_sequences = []
    for seq in tqdm.tqdm(sequences):
        encoded = np.zeros(shape=(len(seq), 4))
        for i, nt in enumerate(seq):
            if nt in ['A', 'C', 'G', 'T', 'P','N']:
                encoded[i, :] = code[nt]
            else:
                encoded[i, :] = code['N']
        encoded_sequences.append(encoded)
    
    return np.array(encoded_sequences)

# Function to prepare train and validation sets
def prepare_train_val_sets_old(gene_response_csv, promoter_terminator_csv, val_chromosome, condition = None,padding_length=20):
    # Read input files
    gene_response_df = pd.read_csv(gene_response_csv, index_col=0)
    promoter_terminator_df = pd.read_csv(promoter_terminator_csv, index_col=0)

    if condition:
        print(f"Filtering genes with a response for {condition}...")
        # Filter genes that have a response for the given condition (0 or 1)
        gene_response_df = gene_response_df[gene_response_df[condition].isin([0, 1])]
        print(f"Number of genes with a response for {condition}: {len(gene_response_df)}")
    # Split into train and validation sets based on the specified validation chromosome
        train_genes = gene_response_df[gene_response_df['Chromosome'] != val_chromosome]
        val_genes = gene_response_df[gene_response_df['Chromosome'] == val_chromosome]

        # Get families present in the train set
        train_families = set(train_genes['family_id'])
        # Filter out validation genes that are in the same family as any training gene
        val_genes_filtered = val_genes[~val_genes['family_id'].isin(train_families)]

        # Print how many validation genes were filtered out
        print(f"Number of validation genes filtered out: {len(val_genes) - len(val_genes_filtered)}")


        # Merge with promoter_terminator_df to get the sequences
        train_set = train_genes.merge(promoter_terminator_df, on='GeneID')
        val_set = val_genes_filtered.merge(promoter_terminator_df, on='GeneID')
        # Check if Chromosome_x and Chromosome_y are the same
        if (train_set['Chromosome_x'] == train_set['Chromosome_y']).all():
            train_set = train_set.rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
        else:
            raise ValueError("Chromosome_x and Chromosome_y do not match in train_set")

        if (val_set['Chromosome_x'] == val_set['Chromosome_y']).all():
            val_set = val_set.rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
        else:
            raise ValueError("Chromosome_x and Chromosome_y do not match in val_set")


        train_families_final = set(train_set['family_id'])
        val_families_final = set(val_set['family_id'])
        # print(f"Gene families in train set: {sorted(train_families_final)}")
        # print(f"Gene families in validation set: {sorted(val_families_final)}")
        print(f"Intersection of gene families: {train_families_final.intersection(val_families_final)}")

        # Print response ratio
        train_response_ratio = train_set[condition].mean()
        val_response_ratio = val_set[condition].mean()
        print(f"Response ratio in train set for {condition}: {train_response_ratio}")
        print(f"Response ratio in validation set for {condition}: {val_response_ratio}")
        print(train_set.columns)
        # Chromosomes in train and validation sets
        train_chromosomes = set(train_set['Chromosome'])
        val_chromosomes = set(val_set['Chromosome'])
        print(f"Chromosomes in train set: {train_chromosomes}")
        print(f"Chromosomes in validation set: {val_chromosomes}")
        print(f"Intersection of chromosomes: {train_chromosomes.intersection(val_chromosomes)}")

        # Prepare sequences by concatenating promoter and terminator with padding
        padding = 'P' * padding_length
        print("padding: ", padding_length)
        print("One hot encoding sequences (train)...")
        train_sequences = onehot(train_set['FullPromoterSequence'] + padding + train_set['FullTerminatorSequence'])
        print("train_sequences: ", train_sequences.shape)
        print("One hot encoding sequences (valid)...")
        val_sequences = onehot(val_set['FullPromoterSequence'] + padding + val_set['FullTerminatorSequence'])
        
        print("val_sequences: ", val_sequences.shape)
        train_responses = train_set[condition].values
        print("train_responses: ", train_responses.shape)
        val_responses = val_set[condition].values
        print("val_responses: ", val_responses.shape)

        return train_sequences, train_responses, val_sequences, val_responses
    
    else:
        print("No condition specified, using all genes")
        conditions = gene_response_df.columns[2:]
        print(f"Conditions: {conditions}")

        # Filter genes that have a response for any condition (0 or 1)
        gene_response_df_dict = {condition: gene_response_df[gene_response_df[condition].isin([0, 1])] for condition in conditions}
        print(f"Number of genes with a response for each condition: {[(condition, len(gene_response_df_dict[condition])) for condition in conditions]}")

        #Split into train and validation sets based on the specified validation chromosome
        train_genes_dict = {condition: gene_response_df_dict[condition][gene_response_df_dict[condition]['Chromosome'] != val_chromosome] for condition in conditions}
        val_genes_dict = {condition: gene_response_df_dict[condition][gene_response_df_dict[condition]['Chromosome'] == val_chromosome] for condition in conditions}
        
        print(f"Number of genes in train set for each condition: {[(condition, len(train_genes_dict[condition])) for condition in conditions]}")
        print(f"Number of genes in validation set for each condition: {[(condition, len(val_genes_dict[condition])) for condition in conditions]}")

        # Get families present in the train set
        train_families_dict = {condition: set(train_genes_dict[condition]['family_id']) for condition in conditions}
        print(f"Gene families in train set for each condition: {[(condition, len(train_families_dict[condition])) for condition in conditions]}")

        # Get full set of families present in train set
        train_families_final = set.union(*train_families_dict.values())
        print(f"Gene families in train set: {len(train_families_final)}")

        # Filter out validation genes that are in the same family as any training gene
        val_genes_filtered_dict = {condition: val_genes_dict[condition][~val_genes_dict[condition]['family_id'].isin(train_families_final)] for condition in conditions}
        print(f"Number of validation genes filtered out for each condition: {[(condition, len(val_genes_dict[condition]) - len(val_genes_filtered_dict[condition])) for condition in conditions]}")
        
        #Merge with promoter_terminator_df to get the sequences
        train_set_dict = {condition: train_genes_dict[condition].merge(promoter_terminator_df, on='GeneID') for condition in conditions}
        val_set_dict = {condition: val_genes_filtered_dict[condition].merge(promoter_terminator_df, on='GeneID') for condition in conditions}

        #Get full set of families present in validation set
        val_families_dict = {condition: set(val_set_dict[condition]['family_id']) for condition in conditions}
        print(f"Gene families in validation set for each condition: {[(condition, len(val_families_dict[condition])) for condition in conditions]}")

        #get full set of families present in validation set
        val_families_final = set.union(*val_families_dict.values())
        print(f"Gene families in validation set: {len(val_families_final)}")

        #Print intersection of gene families
        print(f"Intersection of gene families: {train_families_final.intersection(val_families_final)}")

        #Print response ratio
        train_response_ratio_dict = {condition: train_set_dict[condition][condition].mean() for condition in conditions}
        val_response_ratio_dict = {condition: val_set_dict[condition][condition].mean() for condition in conditions}
        print(f"Response ratio in train set for each condition: {train_response_ratio_dict}")
        print(f"Response ratio in validation set for each condition: {val_response_ratio_dict}")

        #Check if Chromosome_x and Chromosome_y are the same
        for condition in conditions:
            if (train_set_dict[condition]['Chromosome_x'] == train_set_dict[condition]['Chromosome_y']).all():
                train_set_dict[condition] = train_set_dict[condition].rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
            else:
                raise ValueError(f"Chromosome_x and Chromosome_y do not match in train_set for {condition}")

            if (val_set_dict[condition]['Chromosome_x'] == val_set_dict[condition]['Chromosome_y']).all():
                val_set_dict[condition] = val_set_dict[condition].rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
            else:
                raise ValueError(f"Chromosome_x and Chromosome_y do not match in val_set for {condition}")

        #Chromosomes in train and validation sets
        train_chromosomes_dict = {condition: set(train_set_dict[condition]['Chromosome']) for condition in conditions}
        val_chromosomes_dict = {condition: set(val_set_dict[condition]['Chromosome']) for condition in conditions}
        print(f"Chromosomes in train set for each condition: {train_chromosomes_dict}")
        print(f"Chromosomes in validation set for each condition: {val_chromosomes_dict}")
        print(f"Intersection of chromosomes: {train_chromosomes_dict[condition].intersection(val_chromosomes_dict[condition])}")

        #Prepare sequences by concatenating promoter and terminator with padding
        padding = 'P' * padding_length
        print("padding: ", padding_length)
        print("One hot encoding sequences (train)...")
        train_sequences_dict = {condition: onehot(train_set_dict[condition]['FullPromoterSequence'] + padding + train_set_dict[condition]['FullTerminatorSequence']) for condition in conditions}
        
        #print shape of train sequences
        print("train_sequences: ", [train_sequences_dict[condition].shape for condition in conditions])
        print("One hot encoding sequences (valid)...")
        val_sequences_dict = {condition: onehot(val_set_dict[condition]['FullPromoterSequence'] + padding + val_set_dict[condition]['FullTerminatorSequence']) for condition in conditions}
        print("val_sequences: ", [val_sequences_dict[condition].shape for condition in conditions])

        #Get responses
        train_responses_dict = {condition: train_set_dict[condition][condition].values for condition in conditions}
        print("train_responses: ", [train_responses_dict[condition].shape for condition in conditions])
        val_responses_dict = {condition: val_set_dict[condition][condition].values for condition in conditions}
        print("val_responses: ", [val_responses_dict[condition].shape for condition in conditions])

        #conditional embedding
        treatment_mapping = {
            # "A": np.array([0, 0, 0]), # Because this one is not really used. It is used to compare.
            "MeJA": np.array([1, 0, 0]),
            "SA": np.array([0, 1, 0]),
            "SA + MeJA": np.array([1, 1, 0]),
            "ABA": np.array([0, 0, 1]),
            "ABA + MeJA": np.array([1, 0, 1]),
        }
        #create array of conditional embeddings
        train_conditions = np.concatenate([np.tile(treatment_mapping[condition], (len(train_sequences_dict[condition]), 1)) for condition in conditions], axis=0)
        
        #concatenate all the sequences and responses
        train_sequences = np.concatenate([train_sequences_dict[condition] for condition in conditions], axis=0)
        train_responses = np.concatenate([train_responses_dict[condition] for condition in conditions], axis=0)

        #create array of conditional embeddings
        val_conditions = np.concatenate([np.tile(treatment_mapping[condition], (len(val_sequences_dict[condition]), 1)) for condition in conditions], axis=0)
        val_sequences = np.concatenate([val_sequences_dict[condition] for condition in conditions], axis=0)
        val_responses = np.concatenate([val_responses_dict[condition] for condition in conditions], axis=0)

        return train_sequences, train_conditions, train_responses, val_sequences,val_conditions, val_responses

def prepare_train_val_sets(gene_response_csv, promoter_terminator_csv, val_chromosome, condition=None, padding_length=20, full_data=False):
    # Read input files
    gene_response_df = pd.read_csv(gene_response_csv, index_col=0)
    promoter_terminator_df = pd.read_csv(promoter_terminator_csv, index_col=0)

    if condition:
        print(f"Filtering genes with a response for {condition}...")
        # Filter genes that have a response for the given condition (0 or 1)
        gene_response_df = gene_response_df[gene_response_df[condition].isin([0, 1])]
        print(f"Number of genes with a response for {condition}: {len(gene_response_df)}")

        if full_data:
            print("Full data flag activated, using all data for training...")
            train_set = gene_response_df.merge(promoter_terminator_df, on='GeneID')
            padding = 'P' * padding_length
            train_sequences = onehot(train_set['FullPromoterSequence'] + padding + train_set['FullTerminatorSequence'])
            train_responses = train_set[condition].values
            return train_sequences, train_responses, None, None

        # Split into train and validation sets based on the specified validation chromosome
        train_genes = gene_response_df[gene_response_df['Chromosome'] != val_chromosome]
        val_genes = gene_response_df[gene_response_df['Chromosome'] == val_chromosome]

        # Get families present in the train set
        train_families = set(train_genes['family_id'])
        # Filter out validation genes that are in the same family as any training gene
        val_genes_filtered = val_genes[~val_genes['family_id'].isin(train_families)]

        # Print how many validation genes were filtered out
        print(f"Number of validation genes filtered out: {len(val_genes) - len(val_genes_filtered)}")

        # Merge with promoter_terminator_df to get the sequences
        train_set = train_genes.merge(promoter_terminator_df, on='GeneID')
        val_set = val_genes_filtered.merge(promoter_terminator_df, on='GeneID')

        # Check if Chromosome_x and Chromosome_y are the same
        if (train_set['Chromosome_x'] == train_set['Chromosome_y']).all():
            train_set = train_set.rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
        else:
            raise ValueError("Chromosome_x and Chromosome_y do not match in train_set")

        if (val_set['Chromosome_x'] == val_set['Chromosome_y']).all():
            val_set = val_set.rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
        else:
            raise ValueError("Chromosome_x and Chromosome_y do not match in val_set")

        train_families_final = set(train_set['family_id'])
        val_families_final = set(val_set['family_id'])
        print(f"Intersection of gene families: {train_families_final.intersection(val_families_final)}")

        # Print response ratio
        train_response_ratio = train_set[condition].mean()
        val_response_ratio = val_set[condition].mean()
        print(f"Response ratio in train set for {condition}: {train_response_ratio}")
        print(f"Response ratio in validation set for {condition}: {val_response_ratio}")
        print(train_set.columns)

        # Chromosomes in train and validation sets
        train_chromosomes = set(train_set['Chromosome'])
        val_chromosomes = set(val_set['Chromosome'])
        print(f"Chromosomes in train set: {train_chromosomes}")
        print(f"Chromosomes in validation set: {val_chromosomes}")
        print(f"Intersection of chromosomes: {train_chromosomes.intersection(val_chromosomes)}")

        # Prepare sequences by concatenating promoter and terminator with padding
        padding = 'P' * padding_length
        print("padding: ", padding_length)
        print("One hot encoding sequences (train)...")
        train_sequences = onehot(train_set['FullPromoterSequence'] + padding + train_set['FullTerminatorSequence'])
        print("train_sequences: ", train_sequences.shape)
        print("One hot encoding sequences (valid)...")
        val_sequences = onehot(val_set['FullPromoterSequence'] + padding + val_set['FullTerminatorSequence'])
        print("val_sequences: ", val_sequences.shape)
        train_responses = train_set[condition].values
        print("train_responses: ", train_responses.shape)
        val_responses = val_set[condition].values
        print("val_responses: ", val_responses.shape)

        return train_sequences, train_responses, val_sequences, val_responses

    else:
        print("No condition specified, using all genes")
        conditions = gene_response_df.columns[2:]
        print(f"Conditions: {conditions}")

        if full_data:
            print("Full data flag activated, using all data for training...")
            gene_response_df = gene_response_df.melt(id_vars=['GeneID', 'Chromosome', 'family_id'], value_vars=conditions, var_name='condition', value_name='response')
            gene_response_df = gene_response_df[gene_response_df['response'].isin([0, 1])]
            train_set = gene_response_df.merge(promoter_terminator_df, on='GeneID')
            padding = 'P' * padding_length
            train_sequences = onehot(train_set['FullPromoterSequence'] + padding + train_set['FullTerminatorSequence'])
            train_conditions = np.array([treatment_mapping[cond] for cond in train_set['condition']])
            train_responses = train_set['response'].values
            return train_sequences, train_conditions, train_responses, None, None, None

        # Filter genes that have a response for any condition (0 or 1)
        gene_response_df_dict = {condition: gene_response_df[gene_response_df[condition].isin([0, 1])] for condition in conditions}
        print(f"Number of genes with a response for each condition: {[(condition, len(gene_response_df_dict[condition])) for condition in conditions]}")

        # Split into train and validation sets based on the specified validation chromosome
        train_genes_dict = {condition: gene_response_df_dict[condition][gene_response_df_dict[condition]['Chromosome'] != val_chromosome] for condition in conditions}
        val_genes_dict = {condition: gene_response_df_dict[condition][gene_response_df_dict[condition]['Chromosome'] == val_chromosome] for condition in conditions}

        print(f"Number of genes in train set for each condition: {[(condition, len(train_genes_dict[condition])) for condition in conditions]}")
        print(f"Number of genes in validation set for each condition: {[(condition, len(val_genes_dict[condition])) for condition in conditions]}")

        # Get families present in the train set
        train_families_dict = {condition: set(train_genes_dict[condition]['family_id']) for condition in conditions}
        print(f"Gene families in train set for each condition: {[(condition, len(train_families_dict[condition])) for condition in conditions]}")

        # Get full set of families present in train set
        train_families_final = set.union(*train_families_dict.values())
        print(f"Gene families in train set: {len(train_families_final)}")

        # Filter out validation genes that are in the same family as any training gene
        val_genes_filtered_dict = {condition: val_genes_dict[condition][~val_genes_dict[condition]['family_id'].isin(train_families_final)] for condition in conditions}
        print(f"Number of validation genes filtered out for each condition: {[(condition, len(val_genes_dict[condition]) - len(val_genes_filtered_dict[condition])) for condition in conditions]}")

        # Merge with promoter_terminator_df to get the sequences
        train_set_dict = {condition: train_genes_dict[condition].merge(promoter_terminator_df, on='GeneID') for condition in conditions}
        val_set_dict = {condition: val_genes_filtered_dict[condition].merge(promoter_terminator_df, on='GeneID') for condition in conditions}

        # Get full set of families present in validation set
        val_families_dict = {condition: set(val_set_dict[condition]['family_id']) for condition in conditions}
        print(f"Gene families in validation set for each condition: {[(condition, len(val_families_dict[condition])) for condition in conditions]}")

        # Get full set of families present in validation set
        val_families_final = set.union(*val_families_dict.values())
        print(f"Gene families in validation set: {len(val_families_final)}")

        # Print intersection of gene families
        print(f"Intersection of gene families: {train_families_final.intersection(val_families_final)}")

        # Print response ratio
        train_response_ratio_dict = {condition: train_set_dict[condition][condition].mean() for condition in conditions}
        val_response_ratio_dict = {condition: val_set_dict[condition][condition].mean() for condition in conditions}
        print(f"Response ratio in train set for each condition: {train_response_ratio_dict}")
        print(f"Response ratio in validation set for each condition: {val_response_ratio_dict}")

        # Check if Chromosome_x and Chromosome_y are the same
        for condition in conditions:
            if (train_set_dict[condition]['Chromosome_x'] == train_set_dict[condition]['Chromosome_y']).all():
                train_set_dict[condition] = train_set_dict[condition].rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
            else:
                raise ValueError(f"Chromosome_x and Chromosome_y do not match in train_set for {condition}")

            if (val_set_dict[condition]['Chromosome_x'] == val_set_dict[condition]['Chromosome_y']).all():
                val_set_dict[condition] = val_set_dict[condition].rename(columns={'Chromosome_x': 'Chromosome'}).drop(columns=['Chromosome_y'])
            else:
                raise ValueError(f"Chromosome_x and Chromosome_y do not match in val_set for {condition}")

        # Chromosomes in train and validation sets
        train_chromosomes_dict = {condition: set(train_set_dict[condition]['Chromosome']) for condition in conditions}
        val_chromosomes_dict = {condition: set(val_set_dict[condition]['Chromosome']) for condition in conditions}
        print(f"Chromosomes in train set for each condition: {train_chromosomes_dict}")
        print(f"Chromosomes in validation set for each condition: {val_chromosomes_dict}")
        print(f"Intersection of chromosomes: {train_chromosomes_dict[condition].intersection(val_chromosomes_dict[condition])}")

        # Prepare sequences by concatenating promoter and terminator with padding
        padding = 'P' * padding_length
        print("padding: ", padding_length)
        print("One hot encoding sequences (train)...")
        train_sequences_dict = {condition: onehot(train_set_dict[condition]['FullPromoterSequence'] + padding + train_set_dict[condition]['FullTerminatorSequence']) for condition in conditions}

        # Print shape of train sequences
        print("train_sequences: ", [train_sequences_dict[condition].shape for condition in conditions])
        print("One hot encoding sequences (valid)...")
        val_sequences_dict = {condition: onehot(val_set_dict[condition]['FullPromoterSequence'] + padding + val_set_dict[condition]['FullTerminatorSequence']) for condition in conditions}
        print("val_sequences: ", [val_sequences_dict[condition].shape for condition in conditions])

        # Get responses
        train_responses_dict = {condition: train_set_dict[condition][condition].values for condition in conditions}
        print("train_responses: ", [train_responses_dict[condition].shape for condition in conditions])
        val_responses_dict = {condition: val_set_dict[condition][condition].values for condition in conditions}
        print("val_responses: ", [val_responses_dict[condition].shape for condition in conditions])

        # Conditional embedding
        treatment_mapping = {
            "MeJA": np.array([1, 0, 0]),
            "SA": np.array([0, 1, 0]),
            "SA + MeJA": np.array([1, 1, 0]),
            "ABA": np.array([0, 0, 1]),
            "ABA + MeJA": np.array([1, 0, 1]),
        }

        # Create array of conditional embeddings
        train_conditions = np.concatenate([np.tile(treatment_mapping[condition], (len(train_sequences_dict[condition]), 1)) for condition in conditions], axis=0)

        # Concatenate all the sequences and responses
        train_sequences = np.concatenate([train_sequences_dict[condition] for condition in conditions], axis=0)
        train_responses = np.concatenate([train_responses_dict[condition] for condition in conditions], axis=0)

        # Create array of conditional embeddings
        val_conditions = np.concatenate([np.tile(treatment_mapping[condition], (len(val_sequences_dict[condition]), 1)) for condition in conditions], axis=0)
        val_sequences = np.concatenate([val_sequences_dict[condition] for condition in conditions], axis=0)
        val_responses = np.concatenate([val_responses_dict[condition] for condition in conditions], axis=0)

        return train_sequences, train_conditions, train_responses, val_sequences, val_conditions, val_responses

    
    




# Define the CNN model in PyTorch
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=8, padding=4)
        self.pool1 = nn.MaxPool1d(8, padding=4)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8, padding=4)
        self.pool2 = nn.MaxPool1d(8, padding=4)
        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv1d(128, 64, kernel_size=8, padding=4)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=8, padding=4)
        self.pool3 = nn.MaxPool1d(8, padding=4)
        self.dropout3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(448, 128)  # Adjust the input size here
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
# Define the CNN model in PyTorch
class ConvNet_cond(nn.Module):
    def __init__(self, dropout=0.25, end_size=112, embedding_size=3, batch_norm=False):
        super(ConvNet_cond, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv1d(4, 64, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(64) if batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(64) if batch_norm else nn.Identity()
        
        self.pool1 = nn.MaxPool1d(8, padding=4)
        self.dropout1 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(128) if batch_norm else nn.Identity()
        
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8, padding=4)
        self.bn4 = nn.BatchNorm1d(128) if batch_norm else nn.Identity()
        
        self.pool2 = nn.MaxPool1d(8, padding=4)
        self.dropout2 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(128, 64, kernel_size=8, padding=4)
        self.bn5 = nn.BatchNorm1d(64) if batch_norm else nn.Identity()
        
        self.conv6 = nn.Conv1d(64, 64, kernel_size=8, padding=4)
        self.bn6 = nn.BatchNorm1d(64) if batch_norm else nn.Identity()
        
        self.pool3 = nn.MaxPool1d(8, padding=4)
        self.dropout3 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(448 + end_size, 128)  # Adjust the input size here
        self.bn_fc1 = nn.BatchNorm1d(128) if batch_norm else nn.Identity()
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64) if batch_norm else nn.Identity()
        
        self.fc3 = nn.Linear(64, 1)

        self.fc_z = nn.Linear(embedding_size, end_size)
        self.dropout_z = nn.Dropout(dropout)

    def forward(self, x, z):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        z = self.fc_z(z)
        z = torch.relu(z)
        z = self.dropout_z(z)

        x = self.flatten(x)
        x = torch.cat((x, z), 1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout3(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x


# Example usage
if __name__ == "__main__":
    # For unconditional training
    # train_seqs, train_resps, val_seqs, val_resps = prepare_train_val_sets(
    #     gene_response_csv='Data/Processed/filtered_over_under_expressed.csv',
    #     promoter_terminator_csv='Data/Processed/promoter_terminator_sequences.csv',
    #     condition='Mock',
    #     val_chromosome='Chr1'
    # )
    #For conditional training
    train_seqs, train_cond, train_resps, val_seqs,val_cond, val_resps = prepare_train_val_sets(
        gene_response_csv='Data/Processed/filtered_over_under_expressed.csv',
        promoter_terminator_csv='Data/Processed/promoter_terminator_sequences.csv',
        val_chromosome='Chr1'
    )
    #Save the data
    np.save('train_seqs.npy', train_seqs)
    np.save('train_cond.npy', train_cond)
    np.save('train_resps.npy', train_resps)
    np.save('val_seqs.npy', val_seqs)
    np.save('val_cond.npy', val_cond)
    np.save('val_resps.npy', val_resps)

    print(f"Train sequences shape: {train_seqs.shape}")
    print(f"Train responses shape: {train_resps.shape}")
    print(f"Validation sequences shape: {val_seqs.shape}")
    print(f"Validation responses shape: {val_resps.shape}")
    print(f"example train sequence: {train_seqs[0][:10]}")