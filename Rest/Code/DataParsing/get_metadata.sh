wget "https://plantregmap.gao-lab.org/download_ftp.php?filepath=08-download/Arabidopsis_thaliana/binding/regulation_merged_Ath.txt" -O Data/RAW/regulation_merged_Ath.txt

awk -F'\t' 'BEGIN { printf "%s\t%s\n", "TF", "Regulated Gene" } {printf "%s\t%s\n", $1, $3}' Data/RAW/regulation_merged_Ath.txt > Data/RAW/TF_mapping.txt
