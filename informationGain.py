"""
02_information_gain.py
----------------------
Demonstrates Information Gain calculation independently.
"""

import math
from collections import Counter


def entropy(labels):
    total = len(labels)

    if total == 0:
        return 0

    counts = Counter(labels)

    ent = 0.0

    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent


def information_gain(data, attribute, target):

    total = len(data)

    # Entropy before splitting
    labels = [row[target] for row in data]
    total_entropy = entropy(labels)

    # Create subsets
    subsets = {}

    for row in data:

        value = row[attribute]

        if value not in subsets:
            subsets[value] = []

        subsets[value].append(row)

    weighted_entropy = 0.0

    print(f"Splitting on '{attribute}'")
    print("-------------------------")

    for value, subset in subsets.items():

        subset_labels = [row[target] for row in subset]

        subset_entropy = entropy(subset_labels)

        weight = len(subset) / total

        weighted_entropy += weight * subset_entropy

        print(
            f"{value:10s} : "
            f"{Counter(subset_labels)}, "
            f"Entropy = {subset_entropy:.3f}, "
            f"Weight = {weight:.3f}"
        )

    gain = total_entropy - weighted_entropy

    print()
    print(f"Total Entropy       = {total_entropy:.3f}")
    print(f"Weighted Entropy    = {weighted_entropy:.3f}")
    print(f"Information Gain    = {gain:.3f}")

    return gain

def gini_impurity(labels):

    #labels = [row["Loan"] for row in dataset]

    total = len(labels)

    if total == 0:
        return 0

    counts = Counter(labels)
    gini = 1.00

    for labels , count in counts.items():
        p = count / total

        gini -= p**2
    return gini

def split_info(data,attribute):

    total = len(data)

    subset = {}

    for row in data:

        value = row[attribute]  # attribut = poor, good

        if value not in subset:

            subset[value] = []

        subset[value].append(row)

    split = 0.00

    for sub in subset.values():

        p = len(sub) / total


        split -= p * math.log2(p)

    return split




# Independent dataset
dataset = [
    {"Credit Rating": "Excellent", "Loan": "Yes"},
    {"Credit Rating": "Excellent", "Loan": "Yes"},
    {"Credit Rating": "Excellent", "Loan": "Yes"},
    {"Credit Rating": "Excellent", "Loan": "No"},

    {"Credit Rating": "Good", "Loan": "Yes"},
    {"Credit Rating": "Good", "Loan": "No"},
    {"Credit Rating": "Good", "Loan": "Yes"},
    {"Credit Rating": "Good", "Loan": "No"},

    {"Credit Rating": "Poor", "Loan": "No"},
    {"Credit Rating": "Poor", "Loan": "No"},
    {"Credit Rating": "Poor", "Loan": "Yes"},
    {"Credit Rating": "Poor", "Loan": "No"}
]


information_gain(
    dataset,
    "Credit Rating",
    "Loan"
)
labels = [row["Loan"] for row in dataset]

result = gini_impurity(labels)
print("gini impurity:",result)

ans = split_info(dataset,"Credit Rating")
print("split info :",ans)