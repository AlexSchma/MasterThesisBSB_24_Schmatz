import random
import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import chi2_contingency
#from Bio.Seq import Seq
import pandas as pd
import os
import seaborn as sns
def percent_with_motifs(nuc_seqs,motifs):
    found = 0
    for nuc_seq in nuc_seqs:
        if (motifs[0] in nuc_seq or motifs[1] in nuc_seq or motifs[2] in nuc_seq):
            found += 1 
    return found/len(nuc_seqs)
def generate_random_sequence(length=100):
    bases = "ACGT"
    return ''.join(random.choice(bases) for _ in range(length))

def find_motif_positions(dna, motifs):
    for motif in motifs:
        pos = dna.find(motif)
        while pos != -1:
            print(f"Motif '{motif}' found at position {pos}")
            pos = dna.find(motif, pos + 1)

# Function to count motif occurrences and co-occurrences in a list of sequences
def count_motifs(sequences, motifs):
    motif_counts = {motif: 0 for motif in motifs}
    co_occurrence_counts = {(motif1, motif2): 0 for motif1 in motifs for motif2 in motifs if motif1 != motif2}

    for seq in sequences:
        found_motifs = []

        for motif in motifs:
            count = seq.count(motif)
            if count > 0:
                motif_counts[motif] += count
                found_motifs.append(motif)

        for i in range(len(found_motifs)):
            for j in range(i + 1, len(found_motifs)):
                co_occurrence_counts[(found_motifs[i], found_motifs[j])] += 1
                co_occurrence_counts[(found_motifs[j], found_motifs[i])] += 1

    return motif_counts, co_occurrence_counts
def count_motifs_and_plot(sequences, motifs):
    motif_counts_per_sequence = {motif: [] for motif in motifs}

    for seq in sequences:
        for motif in motifs:
            count = seq.count(motif)
            motif_counts_per_sequence[motif].append(count)

    # Plotting histograms for each motif
    for motif, counts in motif_counts_per_sequence.items():
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=range(min(counts), max(counts) + 2), align='left', color='skyblue', edgecolor='black')
        plt.title(f"Histogram of {motif} occurrences per sequence")
        plt.xlabel('Occurrences')
        plt.ylabel('Number of Sequences')
        plt.xticks(range(min(counts), max(counts) + 1))
        plt.grid(axis='y', alpha=0.75)
        plt.show()

def average_motif_occurrences(motifs, nuc_seqs_per_model):
    results = {}

    for model_name, conditions_data in nuc_seqs_per_model.items():
        for condition, nuc_seqs in conditions_data.items():
            total_counts = {motif: 0 for motif in motifs}
            for seq in nuc_seqs:
                for motif in motifs:
                    total_counts[motif] += seq.count(motif)
            avg_counts = {motif: count / len(nuc_seqs) for motif, count in total_counts.items()}
            results.setdefault(model_name, {})[condition] = avg_counts

    return results

def plot_motif_occurrences(results, motifs, results_dir):
    for model_name, conditions_data in results.items():
        df = pd.DataFrame.from_dict(conditions_data, orient='index', columns=motifs)
        df.plot(kind='bar', figsize=(10, 6))
        plt.title(f'Average Motif Occurrences per Sequence for {model_name}')
        plt.ylabel('Average Occurrences')
        plt.xlabel('Condition')
        plt.xticks(rotation=0)
        plt.legend(title='Motif')
        plot_filename = f'{model_name}_motif_occurrences.png'
        plt.savefig(os.path.join(results_dir, plot_filename))
        plt.close()
def plot_roc_curves(Base_model_name,roc_data,results_dir):
    plt.figure()
    for model_name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plot_filename = f'{Base_model_name}_roc_curves.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()

def plot_accuracy_bar(Base_model_name,accuracies,results_dir):
    plt.figure()
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plot_filename = f'{Base_model_name}_accuracy_bar.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()

def aggregate_motif_occurrences(results, motifs):
    # Initialize a dictionary to hold the aggregated results
    aggregated_results = {condition: {motif: [] for motif in motifs} for condition in ['0', '1', '2']}

    # Aggregate the data
    for _, conditions_data in results.items():
        for condition, motif_data in conditions_data.items():
            condition_key = str(int(float(condition)))
            for motif, occurrence in motif_data.items():
                aggregated_results[condition_key][motif].append(occurrence)

    # Calculate the mean occurrence of each motif for each condition across all models
    for condition in aggregated_results:
        for motif in aggregated_results[condition]:
            aggregated_results[condition][motif] = np.mean(aggregated_results[condition][motif])

    return aggregated_results

def plot_aggregated_motif_occurrences(Base_model_name,aggregated_results, results_dir):
    # Convert aggregated results to a DataFrame for easy plotting
    df = pd.DataFrame(aggregated_results).T  # Transpose to have conditions as rows

    # Plotting
    ax = df.plot(kind='bar', figsize=(12, 8))
    plt.title('Average Motif Occurrences per Condition Across All Models')
    plt.ylabel('Average Occurrence')
    plt.xlabel('Condition')
    plt.xticks(rotation=0)
    plt.legend(title='Motif')
    plt.tight_layout()

    # Saving the plot
    plot_filename = f'{Base_model_name}_aggregated_motif_occurrences.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()


def calculate_preference_scores(results, motifs):
    preference_scores = {motif: {str(cond): [] for cond in range(len(motifs))} for motif in motifs}

    for _, conditions_data in results.items():
        for cond in range(len(motifs)):
            cond_key = str(float(cond)) if str(cond) not in conditions_data else str(cond)
            total_occurrences = sum(conditions_data[cond_key].get(motif, 0) for motif in motifs)

            for motif in motifs:
                motif_occurrence = conditions_data[cond_key].get(motif, 0)
                preference_score = motif_occurrence / total_occurrences if total_occurrences > 0 else 0
                preference_scores[motif][str(cond)].append(preference_score)

    return preference_scores
def aggregate_scores_across_models(preference_scores, motifs):
    aggregated_scores = {motif: {cond: np.mean(scores) if scores else 0 
                                 for cond, scores in cond_scores.items()} 
                         for motif, cond_scores in preference_scores.items()}

    return aggregated_scores
def plot_aggregated_preference_scores(aggregated_scores, results_dir,Base_model_name):
    # Convert aggregated scores to a DataFrame for plotting
    df = pd.DataFrame(aggregated_scores).T  # Transpose to have motifs as rows

    # Plotting
    ax = df.plot(kind='bar', figsize=(12, 8), width=0.8)
    plt.title('Aggregated Preference Scores for Motifs Across All Models')
    plt.ylabel('Preference Score')
    plt.xlabel('Motif')
    plt.xticks(rotation=0)
    plt.legend(title='Condition')
    plt.tight_layout()

    # Saving the plot
    plot_filename = f'{Base_model_name}_aggregated_preference_scores.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()


def calculate_relative_increases(results, motifs):
    relative_increases = {motif: [] for motif in motifs}  # Store relative increases for each motif

    for _, conditions_data in results.items():
        for cond, motif_data in conditions_data.items():
            # Convert the condition to an integer for direct comparison
            cond = int(float(cond))
            # For each motif, calculate the relative increase in its corresponding condition
            for motif_index, motif in enumerate(motifs):
                if motif_index == cond:  # Matching condition and motif
                    # Average occurrence of the motif under other conditions
                    avg_other_conditions = np.mean([v[motif] for k, v in conditions_data.items() if int(float(k)) != cond])
                    # Relative increase = occurrence under matching condition - average occurrence under other conditions
                    relative_increase = motif_data[motif] - avg_other_conditions
                    relative_increases[motif].append(relative_increase)

    # Aggregate the relative increases across all models
    aggregated_increases = {motif: np.mean(increases) for motif, increases in relative_increases.items()}
    return aggregated_increases

def plot_relative_increases(aggregated_increases, results_dir,Base_model_name):
    # Convert the aggregated increases to a DataFrame for easy plotting
    df = pd.DataFrame(list(aggregated_increases.items()), columns=['Motif', 'Relative Increase'])

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Motif', y='Relative Increase', data=df)
    plt.title('Relative Increase in Motif Occurrences Under Matching Conditions')
    plt.ylabel('Average Relative Increase')
    plt.xlabel('Motif')

    # Saving the plot
    plot_filename = f'{Base_model_name}_relative_increase_in_motif_occurrences.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()
