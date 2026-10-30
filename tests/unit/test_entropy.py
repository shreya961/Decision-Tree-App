import math
import pytest
from decision_tree_ai.criteria.classification import (
    entropy,
    gini_impurity
)
def test_entropy_equal_classes():
    assert entropy([0.5,0.5]) == 1.0

def test_entropy_pure_class():
    assert entropy([1.0,0.0]) == 0.0

def test_entropy_unequal_classes():
    assert math.isclose ( entropy([0.75,0.25]),0.8112781244591328 )

def test_invalid_probabilities():
    with pytest.raises(ValueError):
        entropy([0.7,0.5])

def test_entropy_single_class():
    assert entropy([1.0]) == 0.0

def test_entropy_ignore_zero_probabilities():
    assert entropy([1.0,0.0]) == 0.0

def test_rejects_empty_probabilities():
    with pytest.raises(ValueError, match='cannot be empty'):
        entropy([])

def test_entropy_rejects_negative_probability():
    with pytest.raises(ValueError,match='between 0 and 1'):
        entropy([1.1,-0.1])
