"""
03_loan_dataset.py
------------------
Displays and analyzes the Loan Approval dataset independently.
"""

from collections import Counter


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


print("LOAN APPROVAL DATASET")
print("=====================")

print(f"Total Records: {len(dataset)}")
print()


print("Dataset:")
print("--------")

for i, row in enumerate(dataset, start=1):

    print(
        f"{i:2d}. "
        f"Credit Rating = {row['Credit Rating']:10s} | "
        f"Loan = {row['Loan']}"
    )


print()
print("Credit Rating Distribution")
print("--------------------------")

ratings = [
    row["Credit Rating"]
    for row in dataset
]

rating_counts = Counter(ratings)

for rating, count in rating_counts.items():

    print(
        f"{rating:10s} = {count} records"
    )


print()
print("Loan Distribution")
print("-----------------")

loans = [
    row["Loan"]
    for row in dataset
]

loan_counts = Counter(loans)

for loan, count in loan_counts.items():

    print(
        f"{loan:5s} = {count} records"
    )