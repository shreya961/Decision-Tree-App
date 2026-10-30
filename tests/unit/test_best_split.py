
import pytest

from decision_tree_ai.core.splitter import best_split




def test_best_split_selects_perfect_feature():
    X = [
        {"assignment": "done", "attendance": "high"},
        {"assignment": "done", "attendance": "low"},
        {"assignment": "missing", "attendance": "low"},
        {"assignment": "missing", "attendance": "low"},
    ]

    y = ["pass", "pass", "fail", "fail"]

    result = best_split(X, y)

    assert result is not None
    assert result.feature == "assignment"
    assert result.information_gain == pytest.approx(1.0)


def test_best_split_returns_none_when_no_feature_can_split():
    X = [
        {"attendance": "high"},
        {"attendance": "high"},
        {"attendance": "high"},
    ]

    y = ["pass", "fail", "pass"]

    assert best_split(X, y) is None


def test_best_split_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        best_split(
            [
                {"assignment": "done"},
                {"assignment": "missing"}
            ],
            ["pass"]
        )
