"""
01_entropy.py
-------------
Demonstrates Entropy calculation independently.
"""

import math
from collections import Counter


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


# Example dataset
labels = ["Yes", "Yes", "Yes", "No", "No", "No"]

print("Entropy Calculation")
print("-------------------")

print("Labels:", labels)

result = entropy(labels)

print(f"Entropy = {result:.3f}")