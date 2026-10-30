import math
import pytest
from decision_tree_ai.criteria.splitting import information_gain

def test_information_gain_for_example():
    gain = information_gain(
        [0.6,0.4],
        [
            (5,[1.0,0.0]),
            (5,[0.2,0.8])
        ]
    )
    assert math.isclose(gain,0.6099865470109875)

def test_information_gain_when_split_changes_nothing():
    gain = information_gain(
        [0.5, 0.5],
        [
            (5, [0.5, 0.5]),
            (5, [0.5, 0.5]),
        ],
    )

    assert math.isclose(gain, 0.0)


def test_information_gain_rejects_empty_children():
    with pytest.raises(ValueError):
        information_gain([0.5, 0.5], [])

def test_information_gain_rejects_zero_size_group():
    with pytest.raises(ValueError):
        information_gain(
            [0.5, 0.5],
            [
                (0, [0.5, 0.5]),
                (10, [0.5, 0.5]),
            ],
        )

def test_information_gain_rejects_negative_size_group():
    with pytest.raises(ValueError):
        information_gain(
            [0.5, 0.5],
            [
                (-1, [0.5, 0.5]),
                (11, [0.5, 0.5]),
            ],
        )

def test_information_gain_accepts_multiple_child_groups():
    gain = information_gain(
        [0.5, 0.5],
        [
            (4, [1.0, 0.0]),
            (6, [0.0, 1.0]),
        ],
    )

def test_information_gain_rejects_invalid_parent_probabilities():
    with pytest.raises(ValueError):
        information_gain([0.7, 0.5], [(10, [0.5, 0.5])])


def test_information_gain_rejects_invalid_child_probabilities():
    with pytest.raises(ValueError):
        information_gain(
            [0.5, 0.5],
            [(10, [0.7, 0.5])],
        )