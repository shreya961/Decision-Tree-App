from decision_tree_ai.criteria.classification import entropy

def information_gain(parent_prob,child_groups):
    if not child_groups:
        raise ValueError('child group cannot be empty')

    if any(size <= 0 for size, _ in child_groups):
        raise ValueError('group size must be positive')

    total_size = sum(size for size, _ in child_groups)

    parent_entropy = entropy(parent_prob)

    weighted_child_entropy = sum( (size / total_size)*entropy(probabilities)
                                 for size, probabilities in child_groups)

    return parent_entropy - weighted_child_entropy
