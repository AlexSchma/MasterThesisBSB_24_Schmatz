#!/bin/bash
echo "test"
# Check if input file, upstream, downstream, and output directory are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <upstream_bp> <downstream_bp> <output_directory>"
    exit 1
fi

# Assign input parameters to variables
UPSTREAM_BP="$1"
DOWNSTREAM_BP="$2"
OUTPUT_DIR="$3"
echo "test2"
wget -qO temp.gff3 https://www.arabidopsis.org/download_files/Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff
echo "test3"

# Convert GFF3 to BED with upstream and downstream regions
awk -v upstream_bp="$UPSTREAM_BP" -v downstream_bp="$DOWNSTREAM_BP" 'BEGIN{FS = OFS ="\t"} $3=="gene" {
    if($7 == "+") {
        upstream_start = ($4 > upstream_bp) ? $4 - upstream_bp -1 : 0
        downstream_end = ($4 + downstream_bp <= $5) ? $4 + downstream_bp : $5
    } else if($7 == "-") {
        upstream_start = ($5 > downstream_bp) ? $5 - downstream_bp -1 : 0
        downstream_end = $5 + upstream_bp
    }
    split($9, id_array, /[=;]/)
    print $1, upstream_start, downstream_end, id_array[2], ".", $7
}' temp.gff3 > "${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"
echo "test4"
rm temp.gff3

echo "Conversion completed. Output saved to ${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"
echo "test5"
bedtools getfasta -fi Data/RAW/TAIR10_chr_all.fas -bed ${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed -name -fo ${OUTPUT_DIR}/promoters_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.fasta
echo "test6"