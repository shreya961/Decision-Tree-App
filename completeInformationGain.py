"""
04_complete_information_gain.py
--------------------------------
Complete Entropy and Information Gain
calculation for a Loan Approval dataset.
"""

import math
from collections import Counter


# --------------------------------------------------
# STEP 1: Entropy
# --------------------------------------------------

def entropy(labels):

    total = len(labels)

    if total == 0:
        return 0

    counts = Counter(labels)

    ent = 0.0

    for label, count in counts.items():

        p = count / total

        ent -= p * math.log2(p)

    return ent


# --------------------------------------------------
# STEP 2: Information Gain
# --------------------------------------------------

def information_gain(data, attribute, target):

    total = len(data)

    # Target labels
    all_labels = [
        row[target]
        for row in data
    ]

    # Entropy before splitting
    total_entropy = entropy(all_labels)

    # Create subsets
    subsets = {}

    for row in data:

        value = row[attribute]

        if value not in subsets:
            subsets[value] = []

        subsets[value].append(row)

    # Calculate weighted entropy
    weighted_entropy = 0.0

    print()
    print(f"Splitting on: {attribute}")
    print("-------------------------")

    for value, subset in subsets.items():

        subset_labels = [
            row[target]
            for row in subset
        ]

        subset_entropy = entropy(subset_labels)

        weight = len(subset) / total

        weighted_entropy += (
            weight * subset_entropy
        )

        print(
            f"{value:10s} -> "
            f"{Counter(subset_labels)}, "
            f"Entropy = {subset_entropy:.3f}, "
            f"Weight = {weight:.3f}"
        )

    # Information Gain
    gain = (
        total_entropy
        - weighted_entropy
    )

    print()
    print(f"Total Entropy(S)    = {total_entropy:.3f}")
    print(f"Weighted Entropy    = {weighted_entropy:.3f}")
    print(f"Information Gain     = {gain:.3f}")

    return gain


# --------------------------------------------------
# STEP 3: Loan Approval Dataset
# --------------------------------------------------

dataset = [

    {"Credit Rating": "Excellent", "Loan": "Yes"},
    {"Credit Rating": "Excellent", "Loan": "Yes"},
    {"Credit Rating": "Excellent", "Loan": "Yes"},

    {"Credit Rating": "Good", "Loan": "Yes"},
    {"Credit Rating": "Good", "Loan": "No"},
    {"Credit Rating": "Good", "Loan": "Yes"},
    {"Credit Rating": "Good", "Loan": "No"},

    {"Credit Rating": "Poor", "Loan": "No"},
    {"Credit Rating": "Poor", "Loan": "No"},
    {"Credit Rating": "Poor", "Loan": "Yes"},
    {"Credit Rating": "Poor", "Loan": "No"},

    {"Credit Rating": "Excellent", "Loan": "No"}
]


# --------------------------------------------------
# STEP 4: Overall Entropy
# --------------------------------------------------

target = "Loan"

labels = [
    row[target]
    for row in dataset
]

print("INFORMATION GAIN CALCULATOR")
print("===========================")

print()
print(f"Total Records = {len(dataset)}")

print()
print("Loan Distribution")
print("-----------------")

print(Counter(labels))

overall_entropy = entropy(labels)

print()
print(
    f"Overall Entropy(S) = "
    f"{overall_entropy:.3f}"
)


# --------------------------------------------------
# STEP 5: Information Gain
# --------------------------------------------------

gain = information_gain(
    dataset,
    "Credit Rating",
    target
)


# --------------------------------------------------
# STEP 6: Final Result
# --------------------------------------------------

print()
print("===========================")
print("FINAL RESULT")
print("===========================")

print(
    f"Information Gain = {gain:.3f}"
)