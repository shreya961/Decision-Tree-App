import math
from collections import Counter
from typing import Any, Sequence

def entropy(probabilities: Sequence[Any]) -> float:
    """
    Calculate the entropy of a classification dataset or class probabilities.
    """
    if not probabilities:
        raise ValueError("Probabilities cannot be empty.")

    # Check if input is a list of floats (representing probability distributions)
    is_float_list = all(
        isinstance(p, float) and not isinstance(p, bool) and math.isfinite(p)
        for p in probabilities
    )

    if is_float_list:
        if any(p < 0 or p > 1 for p in probabilities):
            raise ValueError("Probabilities must be between 0 and 1.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to 1.0.")
        
        ent = 0.0
        for p in probabilities:
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    # If inputs are raw class labels (strings, integers, etc.), compute from frequencies
    total_count = len(probabilities)
    counts = Counter(probabilities)
    
    ent = 0.0
    for count in counts.values():
        p = count / total_count
        if p > 0:
            ent -= p * math.log2(p)
            
    return ent

def gini_impurity(probabilities: Sequence[Any]) -> float:
    """
    Calculate the Gini impurity of a classification dataset or class probabilities.
    """
    if not probabilities:
        raise ValueError("Probabilities cannot be empty.")

    is_float_list = all(
        isinstance(p, float) and not isinstance(p, bool) and math.isfinite(p)
        for p in probabilities
    )

    if is_float_list:
        if any(p < 0 or p > 1 for p in probabilities):
            raise ValueError("Probabilities must be between 0 and 1.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to 1.0.")
            
        impurity = 1.0
        for p in probabilities:
            impurity -= p ** 2
        return impurity

    total_count = len(probabilities)
    counts = Counter(probabilities)
    
    impurity = 1.0
    for count in counts.values():
        p = count / total_count
        impurity -= p ** 2
        
    return impurity