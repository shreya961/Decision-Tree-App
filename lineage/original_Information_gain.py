"""
Information Gain Calculator for Decision Trees
------------------------------------------------
Implements Entropy and Information Gain from scratch (no ML libraries needed),
and demonstrates it on the "Loan Approval" dataset.
"""

import math
from collections import Counter


def entropy(labels):
    """
    Calculate the entropy of a list of class labels.
    Entropy(S) = - sum( p_i * log2(p_i) )
    """
    total = len(labels)
    if total == 0:
        return 0

    counts = Counter(labels)
    ent = 0.0
    for label, count in counts.items():
        p = count / total
        ent -= p * math.log2(p)
    return ent


def information_gain(data, attribute, target):
    """
    Calculate Information Gain of splitting `data` on `attribute`.

    data: list of dicts, e.g. [{"Credit Rating": "Good", "Loan": "Yes"}, ...]
    attribute: the column name to split on, e.g. "Credit Rating"
    target: the column name of the class label, e.g. "Loan"
    """
    total = len(data)
    all_labels = [row[target] for row in data]
    total_entropy = entropy(all_labels)

    # Group rows by the attribute's value
    subsets = {}
    for row in data:
        value = row[attribute]
        subsets.setdefault(value, []).append(row)

    # Weighted average entropy of the subsets
    weighted_entropy = 0.0
    print(f"\nSplitting on '{attribute}':")
    for value, subset in subsets.items():
        subset_labels = [row[target] for row in subset]
        subset_entropy = entropy(subset_labels)
        weight = len(subset) / total
        weighted_entropy += weight * subset_entropy
        print(f"  {attribute} = {value:10s} -> "
              f"{Counter(subset_labels)}, "
              f"Entropy = {subset_entropy:.3f}, Weight = {weight:.3f}")

    gain = total_entropy - weighted_entropy
    print(f"  Total Entropy(S)       = {total_entropy:.3f}")
    print(f"  Weighted Entropy       = {weighted_entropy:.3f}")
    print(f"  Information Gain       = {gain:.3f}")
    return gain


if __name__ == "__main__":
    # Loan Approval dataset (from the worked example)
    dataset = [
        {"Credit Rating": "Excellent", "Loan": "Yes"},
        {"Credit Rating": "Excellent", "Loan": "Yes"},
        {"Credit Rating": "Excellent", "Loan": "Yes"},
        {"Credit Rating": "Good",      "Loan": "Yes"},
        {"Credit Rating": "Good",      "Loan": "No"},
        {"Credit Rating": "Good",      "Loan": "Yes"},
        {"Credit Rating": "Good",      "Loan": "No"},
        {"Credit Rating": "Poor",      "Loan": "No"},
        {"Credit Rating": "Poor",      "Loan": "No"},
        {"Credit Rating": "Poor",      "Loan": "Yes"},
        {"Credit Rating": "Poor",      "Loan": "No"},
        {"Credit Rating": "Excellent", "Loan": "No"},
    ]

    target_col = "Loan"

    # Overall entropy of the dataset
    labels = [row[target_col] for row in dataset]
    print(f"Overall Entropy(S) = {entropy(labels):.3f}")

    # Information Gain for the "Credit Rating" attribute
    ig = information_gain(dataset, "Credit Rating", target_col)

    # --- To compare multiple attributes, just loop over them ---
    # for attr in ["Credit Rating", "Income Level", "Employment Status"]:
    #     information_gain(dataset, attr, target_col)
