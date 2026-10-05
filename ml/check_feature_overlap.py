from collections import defaultdict, Counter
import csv

pairs = defaultdict(dict)

with open("dataset/ml_dataset.csv", newline="") as file:
    for row in csv.DictReader(file):
        features = {
            key: value
            for key, value in row.items()
            if key not in ("pair_id", "label")
        }

        pairs[row["pair_id"]][row["label"]] = features

same = 0
changes = Counter()

for pair in pairs.values():
    if pair["0"] == pair["1"]:
        same += 1
        continue

    for feature in pair["0"]:
        if pair["0"][feature] != pair["1"][feature]:
            changes[feature] += 1

print("Total pairs:", len(pairs))
print("Identical feature pairs:", same)
print("Percentage:", f"{same / len(pairs):.1%}")

print("\nFeature changes among non-identical pairs:")

for feature, count in changes.most_common():
    print(f"{feature}: {count}")