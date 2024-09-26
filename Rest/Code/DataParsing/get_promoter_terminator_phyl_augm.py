import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import os
import copy

ecotypes = ["An-1","C24","Cvi","Kyo","Sha","Eri","Ler"]
ecotype_main_dir = "Data/RAW/ecotype_data/"
col_0_genome_path = "Data/RAW/TAIR10_chr_all.fas"
col_0_gff_path = "Data/RAW/TAIR10_GFF3_genes.gff"
response_path = "Data/Processed/over_under_expressed.csv"
gene_families_df_path = "Data/Processed/gene_families.csv"

def check_gene_id(ecotypes,ecotype_main_dir,col_0_gff_path,col_0_genome_path,response_path,gene_families_df_path):
    ecotype_fasta_paths = {}
    ecotype_gff_paths = {}

    for ecotype in ecotypes:
        fasta_file = f"{ecotype}.chr.all.v2.0.fasta/{ecotype}.chr.all.v2.0.fasta"
        gff_file = f"{ecotype}.protein-coding.genes.v2.5.2019-10-09.gff3/{ecotype}.protein-coding.genes.v2.5.2019-10-09.gff3"

        ecotype_fasta_paths[ecotype] = os.path.join(ecotype_main_dir, fasta_file)
        ecotype_gff_paths[ecotype] = os.path.join(ecotype_main_dir, gff_file)

    df = pd.read_csv(response_path, index_col=0)

    # Remove the first two letters from each index (gene ID)
    df.index = df.index.str[2:]

    # Convert the index to a list if needed
    gene_ids = df.index.tolist()

    not_found = []
    col_0_gene_pos_dict = {}
    with open(col_0_gff_path) as gff:
        for line in tqdm(gff, desc="Processing GFF3 file"):
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if (parts[2] == "gene" or parts[2] == "transposable_element_gene" or parts[2] == "pseudogene")  and "ID=" in parts[8]:
                gene_id = parts[8].split("ID=")[1].split(";")[0][2:]
                if gene_id in gene_ids:
                    chrom = parts[0]
                    start = int(parts[3])
                    end = int(parts[4])
                    strand = parts[6]
                    col_0_gene_pos_dict[gene_id] = (chrom, start, end, strand)
                else:
                    not_found.append(gene_id)
    
    if len(not_found)!=0:
        print(f"Number of genes not found in the response file: {len(not_found)}")
        print("Gene IDs not found:", not_found)
    else:
        print("All gene IDs found in the response file.")
    
    remaining_gene_id_sets = {}
    ecotype_gene_pos_dict = {}
    for ecotype in ecotypes:
        gene_ids_temp = copy.deepcopy(gene_ids)
        not_found = []
        ecotype_gene_pos_dict[ecotype] = {}
        with open(ecotype_gff_paths[ecotype]) as gff:
            for line in tqdm(gff, desc=f"Processing {ecotype} GFF3 file"):
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if (parts[2] == "gene" or parts[2] == "transposable_element_gene" or parts[2] == "pseudogene")  and "ID=" in parts[8]:
                    gene_id = parts[8].split("ID=")[1].split(";")[0].split("-")[1]
                    if gene_id.startswith("U"):
                        continue
                    if gene_id in gene_ids:
                        chrom = parts[0]
                        start = int(parts[3])
                        end = int(parts[4])
                        strand = parts[6]
                        ecotype_gene_pos_dict[ecotype][gene_id] = (chrom, start, end, strand)
                        gene_ids_temp.remove(gene_id)
                    else:
                        not_found.append(gene_id)
        remaining_gene_id_sets[ecotype] = set(gene_ids_temp)
        if len(not_found)!=0:
            print(f"Number of genes not found in the response file for {ecotype}: {len(not_found)}")
            #print("Gene IDs not found:", not_found)

    print("Genes in response file:", len(gene_ids))
    print("Genes found in the col-0 gff:", len(col_0_gene_pos_dict))
    for ecotype in ecotypes:
        print(f"Genes found in the {ecotype} gff:", len(ecotype_gene_pos_dict[ecotype]))
    for ecotype in ecotypes:
        print(f"Genes not found in the {ecotype} gff:", len(remaining_gene_id_sets[ecotype]))
    

    genes_not_found = set()
    for ecotype in ecotypes:
        genes_not_found.update(remaining_gene_id_sets[ecotype])
    print(f"nr genes not found: {len(genes_not_found)}")

    #genes found in none of the ecotype gffs
    all_sets = remaining_gene_id_sets.values()
    genes_not_found = set.intersection(*all_sets)
    print(f"nr genes not found in any ecotype: {len(genes_not_found)}")
    print(f"genes not found in any ecotype: {genes_not_found}")
    



        
    
              
    










def process_gene_sequences(gff_file, fasta_file, input_csv, output_csv, filtered_output_csv, gene_families_csv, promoter_length=1000, terminator_length=1000, utr5prime_length=500, utr3prime_length=500, debug=False,only_genes=False):
    # Read input CSV file
    df = pd.read_csv(input_csv, index_col=0)
    gene_ids = df.index.tolist()

    # Read the genome sequence from the FASTA file
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    # Read gene families CSV
    gene_families_df = pd.read_csv(gene_families_csv)

    # Function to extract sequences based on annotations
    def extract_sequences(gff_file, gene_ids, genome, promoter_length, terminator_length, utr5prime_length, utr3prime_length, debug):
        sequences_list = []
        i = 0
        with open(gff_file) as gff:
            for line in tqdm(gff, desc="Processing GFF3 file"):
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if only_genes and parts[2] == "gene" and "ID=" in parts[8] or ((parts[2] == "gene" or parts[2] == "transposable_element_gene" or parts[2] == "pseudogene")  and "ID=" in parts[8]):
                    gene_id = parts[8].split("ID=")[1].split(";")[0]
                    if gene_id in gene_ids:
                        chrom = parts[0]
                        start = int(parts[3])
                        end = int(parts[4])
                        strand = parts[6]
                        
                        if strand == "+":
                            promoter_start = max(1, start - promoter_length)
                            promoter_end = start + utr5prime_length - 1
                            terminator_start = end - utr3prime_length + 1
                            terminator_end = end + terminator_length
                        else:
                            promoter_start = end - utr5prime_length + 1
                            promoter_end = end + promoter_length
                            terminator_start = max(1, start - terminator_length)
                            terminator_end = start + utr3prime_length - 1
                        
                        promoter_seq = genome[chrom].seq[promoter_start-1:promoter_end].upper()
                        terminator_seq = genome[chrom].seq[terminator_start-1:terminator_end].upper()
                        
                        if strand == "-":
                            promoter_seq = promoter_seq.reverse_complement()
                            terminator_seq = terminator_seq.reverse_complement()
                        
                        if debug:
                            print("Information of the gene:", gene_id)
                            print("Chromosome:", chrom)
                            print("Promoter start:", promoter_start, "Promoter end:", promoter_end)
                            print("Terminator start:", terminator_start, "Terminator end:", terminator_end)
                            print("Promoter sequence:", promoter_seq)
                            print("Terminator sequence:", terminator_seq)
                            print("_"*50)
                        
                        sequences_list.append({
                            'GeneID': gene_id,
                            'Chromosome': chrom,
                            'FullPromoterSequence': str(promoter_seq),
                            'FullTerminatorSequence': str(terminator_seq)
                        })
                        
                        if debug:
                            i += 1
                            if i == 3:  # Break after processing three genes for debugging
                                break
        return sequences_list

    # Extract sequences
    sequences_list = extract_sequences(gff_file, gene_ids, genome, promoter_length, terminator_length, utr5prime_length, utr3prime_length, debug)

    # Convert list to DataFrame
    sequences_df = pd.DataFrame(sequences_list)

    # Sanity checks
    missing_promoter = sequences_df['FullPromoterSequence'].isnull().sum()
    missing_terminator = sequences_df['FullTerminatorSequence'].isnull().sum()
    promoter_lengths = sequences_df['FullPromoterSequence'].dropna().apply(len)
    terminator_lengths = sequences_df['FullPromoterSequence'].dropna().apply(len)

    print(f"Number of genes with missing promoter sequences: {missing_promoter}")
    print(f"Number of genes with missing terminator sequences: {missing_terminator}")

    # Print counts of unique promoter sequence lengths
    print("Unique promoter sequence lengths and their counts:")
    print(promoter_lengths.value_counts().sort_index())

    # Print counts of unique terminator sequence lengths
    print("Unique terminator sequence lengths and their counts:")
    print(terminator_lengths.value_counts().sort_index())

    # Save the sequences to a new CSV file
    if not debug:
        sequences_df.to_csv(output_csv, index=False)

    # Expected lengths
    promoter_length_expected = promoter_length + utr5prime_length
    terminator_length_expected = terminator_length + utr3prime_length

    # Add length columns for filtering
    sequences_df['PromoterLength'] = sequences_df['FullPromoterSequence'].apply(len)
    sequences_df['TerminatorLength'] = sequences_df['FullTerminatorSequence'].apply(len)

    # Find genes that don't meet the criteria
    invalid_genes_df = sequences_df[
        (sequences_df['PromoterLength'] != promoter_length_expected) | 
        (sequences_df['TerminatorLength'] != terminator_length_expected)
    ]

    # Print number of rows removed
    num_removed = len(invalid_genes_df)
    print(f"Number of genes removed: {num_removed}")

    # If less than 10 genes removed, print their IDs
    if num_removed < 10:
        removed_genes = invalid_genes_df['GeneID'].tolist()
        print("Removed genes:", removed_genes)

    # Filter the DataFrame
    filtered_sequences_df = sequences_df[
        (sequences_df['PromoterLength'] == promoter_length_expected) & 
        (sequences_df['TerminatorLength'] == terminator_length_expected)
    ]

    # Add gene family information
    filtered_sequences_df = filtered_sequences_df.merge(gene_families_df, left_on='GeneID', right_on='gene_id', how='left').drop(columns=['gene_id']).merge(df, left_on='GeneID', right_index=True, how='left')

    # Save the filtered DataFrame to a new CSV file
    #filtered_sequences_df[['GeneID', 'Chromosome', 'family_id','ABA','ABA + MeJA','MeJA','Mock','SA','SA + MeJA']].to_csv(filtered_output_csv, index=False)
    filtered_sequences_df[['GeneID', 'Chromosome', 'family_id','ABA','ABA + MeJA','MeJA','SA','SA + MeJA']].to_csv(filtered_output_csv, index=False)

    # Print a message indicating the file has been saved
    print(f"Filtered gene list saved to {filtered_output_csv}")

# Example usage:
if __name__ == "__main__":
    # process_gene_sequences(
    #     gff_file='Data/RAW/TAIR10_GFF3_genes.gff',
    #     fasta_file='Data/RAW/TAIR10_chr_all.fas',
    #     input_csv='Data/Processed/over_under_expressed.csv',
    #     output_csv='Data/Processed/promoter_terminator_sequences.csv',
    #     filtered_output_csv='Data/Processed/filtered_over_under_expressed.csv',
    #     gene_families_csv='Data/Processed/gene_families.csv',
    #     promoter_length=1000,
    #     terminator_length=1000,
    #     utr5prime_length=500,
    #     utr3prime_length=500,
    #     debug=False
    # )
    check_gene_id(ecotypes,ecotype_main_dir,col_0_gff_path,col_0_genome_path,response_path,gene_families_df_path)
