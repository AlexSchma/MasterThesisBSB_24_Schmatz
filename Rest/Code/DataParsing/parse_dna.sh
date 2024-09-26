#!/bin/bash

# Check if input file, upstream, downstream, upstream TTS, downstream TTS and output directory are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <upstream_bp_TSS> <downstream_bp_TSS> <upstream_bp_TTS> <downstream_bp_TTS> <output_dir>"
    exit 1
fi

# Check if bedtools is installed
if ! command -v bedtools &> /dev/null; then
    echo "bedtools is not installed. Please install bedtools before running this script. See readme"
    exit 1
fi

# Assign input parameters to variables
UPSTREAM_BP="$1"
DOWNSTREAM_BP="$2"
UPSTREAM_BP_TTS="$3"
DOWNSTREAM_BP_TTS="$4"
OUTPUT_DIR="$5"

if ! [ -f ${OUTPUT_DIR}/TAIR10_chr_all.fas ]; then
    
    wget https://www.arabidopsis.org/download_files/Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas.gz -P ${OUTPUT_DIR}/
    gunzip ${OUTPUT_DIR}/TAIR10_chr_all.fas.gz
    rm ${OUTPUT_DIR}/TAIR10_chr_all.fas.gz
fi

if ! [  -f ${OUTPUT_DIR}/TAIR10_GFF3_genes.gff ]; then
    wget https://www.arabidopsis.org/download_files/Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff -P ${OUTPUT_DIR}/

fi

awk '($3 == "chromosome") {print $1, $5 }' "${OUTPUT_DIR}/TAIR10_GFF3_genes.gff" > "${OUTPUT_DIR}/chromosome_lengths.txt"

# Convert GFF3 to BED with upstream and downstream regions
awk -v upstream_bp="$UPSTREAM_BP" -v downstream_bp="$DOWNSTREAM_BP" 'BEGIN{FS = OFS ="\t"} index($3, "gene") {
    if($7 == "+") {
        upstream_start = ($4 > upstream_bp) ? $4 - upstream_bp -1 : 0
        downstream_end = $4 + downstream_bp 
    } else if($7 == "-") {
        upstream_start = ($5 > downstream_bp) ? $5 - downstream_bp -1 : 0
        downstream_end = $5 + upstream_bp 
    } 
    split($9, id_array, /[=;]/)
    print $1, upstream_start, downstream_end, id_array[2], ".", $7
}' ${OUTPUT_DIR}/TAIR10_GFF3_genes.gff > "${OUTPUT_DIR}/tmp_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

# Remove splice variants
awk -F'\t' '$4 !~ /\./' "${OUTPUT_DIR}/tmp_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed" > "${OUTPUT_DIR}/dirty_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

rm "${OUTPUT_DIR}/tmp_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed" 

# Now limit the size to chromosome length to avoid bedtools to skip these requests:
awk 'BEGIN{OFS="\t"} ($3 == "chromosome") {chrom_lengths[$1] = $2} 
     NR==FNR{chrom_lengths[$1]=$2; next} 
     ($1 in chrom_lengths) {if ($3 > chrom_lengths[$1]) $3 = chrom_lengths[$1]; print}' "${OUTPUT_DIR}/chromosome_lengths.txt" "${OUTPUT_DIR}/dirty_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed" > "${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

rm "${OUTPUT_DIR}/dirty_custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

echo "Conversion completed. Output saved to ${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed"

bedtools getfasta -fi ${OUTPUT_DIR}/TAIR10_chr_all.fas \
        -bed ${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.bed\
         -name -fo ${OUTPUT_DIR}/promoters_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.fasta \
         #-s # Force strandedness. If the feature occupies the antisense strand, the sequence will be reverse complemented

echo "Fasta file saved to ${OUTPUT_DIR}/promoters_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TSS.fasta, good to go!"

# Convert GFF3 to BED with upstream and downstream regions
awk -v upstream_bp="$UPSTREAM_BP_TTS" -v downstream_bp="$DOWNSTREAM_BP_TTS" 'BEGIN{FS = OFS ="\t"} index($3, "gene") {
    if($7 == "+") {
        upstream_start = ($5 > upstream_bp) ? $5 - upstream_bp -1 : 0
        downstream_end = $5 + downstream_bp
    } else if($7 == "-") {
        upstream_start = ($4 > downstream_bp) ? $4 - downstream_bp -1 : 0
        downstream_end = $4 + upstream_bp
    } 
    split($9, id_array, /[=;]/)
    print $1, upstream_start, downstream_end, id_array[2], ".", $7
}' ${OUTPUT_DIR}/TAIR10_GFF3_genes.gff > "${OUTPUT_DIR}/tmp_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

# Remove splice variants
awk -F'\t' '$4 !~ /\./' "${OUTPUT_DIR}/tmp_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed" > "${OUTPUT_DIR}/dirty_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

rm "${OUTPUT_DIR}/tmp_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed" 

awk 'BEGIN{OFS="\t"} ($3 == "chromosome") {chrom_lengths[$1] = $2} 
     NR==FNR{chrom_lengths[$1]=$2; next} 
     ($1 in chrom_lengths) {if ($3 > chrom_lengths[$1]) $3 = chrom_lengths[$1]; print}' "${OUTPUT_DIR}/chromosome_lengths.txt" "${OUTPUT_DIR}/dirty_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed" > "${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

rm "${OUTPUT_DIR}/dirty_custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

echo "Conversion completed. Output saved to ${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed"

bedtools getfasta -fi ${OUTPUT_DIR}/TAIR10_chr_all.fas -bed ${OUTPUT_DIR}/custom_promoter_coordinates_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.bed -name -fo ${OUTPUT_DIR}/promoters_${UPSTREAM_BP_TTS}up_${DOWNSTREAM_BP_TTS}down_TTS.fasta #-s

rm "${OUTPUT_DIR}/chromosome_lengths.txt"

echo "Fasta file saved to ${OUTPUT_DIR}/promoters_${UPSTREAM_BP}up_${DOWNSTREAM_BP}down_TTS.fasta, good to go!"


# Now the introns!
if ! [ -f ${OUTPUT_DIR}/TAIR10_GFF3_genes_introns.gff ]; then
agat_sp_add_introns.pl --gff ${OUTPUT_DIR}/TAIR10_GFF3_genes.gff --out ${OUTPUT_DIR}/TAIR10_GFF3_genes_introns.gff
fi

# Now I have my introns coordinates, the problem is that each intron is going to correspnd to a particular splice variant. So this is
# more complex than the previous step. Furthermore, legnth will be variable.

awk 'BEGIN{FS = OFS = "\t"} 
     index($3, "intron") {
         split($9, id_array, /[;]/)
         for(i in id_array) {
             if(index(id_array[i], "Parent=")) {
                 split(id_array[i], parent_array, "=")
                 split(parent_array[2], gene_array, "-")
                 parent_gene = gene_array[1]
             }
         }
         print $1, $4, $5, parent_gene, ".", $7
     }' ${OUTPUT_DIR}/TAIR10_GFF3_genes_introns.gff > "${OUTPUT_DIR}/intron_coordinates_dup.bed"

awk '!a[$0]++' ${OUTPUT_DIR}/intron_coordinates_dup.bed > ${OUTPUT_DIR}/intron_coordinates.bed
rm ${OUTPUT_DIR}/intron_coordinates_dup.bed
awk '$4 ~ /\.1$/' ${OUTPUT_DIR}/intron_coordinates.bed > ${OUTPUT_DIR}/intron_coordinates_only_first_isoform.bed
rm ${OUTPUT_DIR}/intron_coordinates.bed

#rm ${OUTPUT_DIR}/intron_coordinates_dub.bed

bedtools getfasta -fi ${OUTPUT_DIR}/TAIR10_chr_all.fas -bed ${OUTPUT_DIR}/intron_coordinates_only_first_isoform.bed -name -fo ${OUTPUT_DIR}/introns.fasta #-s




