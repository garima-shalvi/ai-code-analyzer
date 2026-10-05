from datasets import load_dataset
from agents.code_understanding import get_target_function, get_function_features
import csv

def clean_code(code):
    return code.replace("\\n", "\n").replace("\\t", "\t")

dataset = load_dataset("OSS-forge/PyResBugs", split="train")

rows = []

for pair_id, row in enumerate(dataset):
    faulty_code = clean_code(row["Faulty Code"])
    fixed_code = clean_code(row["Fault Free Code"])

    faulty_function = get_target_function(
        faulty_code,
        row["Fixed_Method"]
    )

    fixed_function = get_target_function(
        fixed_code,
        row["Fixed_Method"]
    )

    if faulty_function is None or fixed_function is None:
        continue

    faulty_features = get_function_features(faulty_function)
    fixed_features = get_function_features(fixed_function)

    faulty_features.pop("function_name")
    fixed_features.pop("function_name")

    faulty_features["pair_id"] = pair_id
    faulty_features["label"] = 1

    fixed_features["pair_id"] = pair_id
    fixed_features["label"] = 0

    rows.append(faulty_features)
    rows.append(fixed_features)

fieldnames = [
    "pair_id",
    "loop_count",
    "condition_count",
    "call_count",
    "subscript_access_count",
    "max_nesting_depth",
    "max_loop_nesting",
    "is_recursive",
    "function_length",
    "label"
]

with open("dataset/ml_dataset.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Total samples:", len(rows))
print("Buggy samples:", sum(row["label"] == 1 for row in rows))
print("Fixed samples:", sum(row["label"] == 0 for row in rows))
print("Saved to: dataset/ml_dataset.csv")