Decision-Tree-App
A from-scratch Decision Tree library and interactive application in Python, built with a focus on
mathematical correctness, thorough testing, and clear documentation.
Status &amp; Release
 v1.1 Released: 27/27 tests passed. Expands testing, robustness, and package organization.
 v1.0 Tagged (Previous): 16/16 tests passed. Established the immediate foundation covering
core mathematics (entropy, Gini impurity, and information gain) with worked examples.
Table of Contents
 Overview
 Background
 Guiding Principle
 Current Status &amp; Roadmap
 The Mathematics
 Getting Started
 Quality Targets &amp; Testing Standards
 Project Structure
 Deferred Features
 Contributing
 Author
Overview
Decision-Tree-App is developed across two structured divisions:
 Division I: Immediate Foundation ✅ Completed &amp; Tagged v1.0 / v1.1 The core foundation:
correct Entropy, Gini impurity, and Information Gain implementations backed by a robust,
automated test suite and clean package structure.
 Division II: Complete Development ▶ Next A comprehensive, reusable library covering
classification, regression, preprocessing, evaluation, explainability, visualization,
benchmarking, and an interactive application for final release.
Background
The foundation was built on rigorous theoretical study prior to writing code, covering:
 Mathematical fundamentals: Probability, logarithms, and information theory.
 Metrics: Shannon entropy, Gini impurity, and Information Gain.
 Algorithm theory: Decision tree construction, split selection, and recursive tree growing
(ID3).

Because the mathematics was worked out by hand first, the code is rigorously verified against hand-
computed values rather than trusted blindly.
Guiding Principle
Build one reliable engine first. Every interface—including notebook examples and the interactive
application—calls that same core engine and never re-implements an algorithm.
Current Status &amp; Roadmap
Division I: Immediate Foundation (Completed — Tagged v1.1)
 Project root and Python environment verified.
 Package imports working seamlessly from a fresh shell.
 Entropy and Gini impurity implemented and thoroughly tested.
 Information Gain implemented with worked loan-dataset examples.
 Full test suite passing (28/28 unit tests).
Division II: Complete Library Development
1. Classification Foundation: Nodes, best-split selection, recursive ID3, and readable tree
output.
2. Classification API: fit(), predict(), predict_proba(), numerical splits (), hyperparameters
(max_depth, min_samples_split, min_samples_leaf), deterministic tie-breaking, and
classification metrics.
3. Regression Trees: Mean-value leaves, variance-reduction splits, and regression metrics
(MAE, MSE, RMSE, ).
4. Robustness &amp; Preprocessing: Missing-value policies, categorical encoding, type validation,
cost-complexity pruning, reproducible random_state, and unseen-category handling.
5. Research Quality: Benchmarks against scikit-learn, prediction explanations, feature
importance, and tree visualization.
6. Final Release: Streamlit application integration, end-to-end tests, quick-start guide, API
reference, and mathematical documentation.
Application workflow preview: Upload a CSV select target column choose classification or
regression configure settings train view metrics inspect tree explore prediction paths download
results.
The Mathematics
1. Shannon Entropy
Calculated in bits, with the convention :

2. Gini Impurity

3. Information Gain
For a parent set split into subsets :

Reference Validation Values
Class
Probabilities Entropy Gini

Note: Invalid probabilities (negative values, values above 1, or totals other than 1 within a
documented tolerance) raise a ValueError with a clear message. Entropy and Gini share a single
validation helper to ensure consistent input handling.
Getting Started
Installation and Testing (Current v1.1)
Bash
git clone https://github.com/shreya961/Decision-Tree-App.git
cd Decision-Tree-App
pip install -e .
python -m pytest tests/unit -v
Example Usage
Python
from decision_tree_ai.criteria import entropy, gini, information_gain

# Example calculation
ent = entropy([0.5, 0.5])
print(f&quot;Entropy: {ent}&quot;)
(Function names represent the public API and may be refined as the package matures.)
Quality Targets &amp; Testing Standards

 Test Coverage Targets: 200–250+ automated tests long-term; core-engine statement
coverage targeting 85–90% where practical.
 Standards:
o Compare outputs with independently computed, known results.
o Use explicit tolerances for floating-point comparisons (pytest.approx).
o Ensure invalid inputs raise documented exceptions with informative messages.
o Every corrected defect receives a dedicated regression test.

Project Structure
Plaintext
Decision-Tree-App/
├── src/decision_tree_ai/
│ ├── core/
│ ├── criteria/ # entropy, gini, information gain
│ ├── algorithms/ # ID3, classification and regression trees
│ ├── preprocessing/
│ ├── evaluation/
│ ├── explainability/
│ └── visualization/
├── tests/ # unit, integration, end_to_end
├── app/ # Streamlit application
├── examples/
├── notebooks/
├── data/
├── docs/
├── reports/
├── README.md
├── pyproject.toml
├── requirements-dev.txt
├── CONTRIBUTING.md
├── LICENSE
└── .gitignore

Deferred Until After v2.0.0
The following features are out of scope until the final stable release is tagged:
 Random Forests and other ensemble methods
 Gradient-boosted decision trees
 Automated feature engineering
 Distributed training and large-scale optimisation
 Separate production-grade web backend and frontend
 Research on novel split criteria or pruning algorithms
Contributing
Contributions are welcome! Please follow these guidelines:
1. Open one issue per feature or defect.
2. Make small, focused commits.
3. Run the full test suite before opening a pull request.
4. Use pull requests for substantial changes.
Author
Shreya (@shreya961)
