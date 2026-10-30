"""Candidate split generation and evaluation."""
from dataclasses import dataclass
from typing import Any

from decision_tree_ai.criteria.classification import entropy

@dataclass
class SplitResult:
    "stores the result of the best categorical split"
    feature: str
    information_gain: float
    groups: dict[Any, list[int]]

def best_split(
        x: list[dict[str, Any]],
        y: list[Any],
) -> SplitResult | None: 
    """
    Find the categorical feature with the highest information gain.

    X: One dictionary per row, with feature names and values.
    y: Class label for each row.

    The returned groups contain row indices.
    """
    if len(x) != len(y):
        raise ValueError('x and y must contain same number of rows')

    if not x:
        raise ValueError('dataset cannot be empty')

    if not x[0]:
        return None
    
    if any(not row for row in x):
        raise ValueError("Every row must contain features.")

    parent_entropy = entropy(y)
    features = x[0].keys()

    best_feature: str | None = None
    best_info_gain = -1.0
    best_groups: dict[Any, list[int]] = {}
    total_samples = len(y)

    for feature in features:
        groups: dict[Any, list[int]] = {}
        
        for idx, row in enumerate(x):
            val = row.get(feature)
            if val not in groups:
                groups[val] = []
            groups[val].append(idx)

        weighted_entropy = 0.0
        for indices in groups.values():
            subset_y = [y[i] for i in indices]
            weighted_entropy += (len(indices) / total_samples) * entropy(subset_y)

        info_gain = parent_entropy - weighted_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
            best_groups = groups

    if best_feature is None or best_info_gain <= 0.0:
        return None

    return SplitResult(
        feature=best_feature,
        information_gain=best_info_gain,
        groups=best_groups
    )