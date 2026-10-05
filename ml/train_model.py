import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
with open("dataset/ml_dataset.csv", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

pair_ids = sorted(set(row["pair_id"] for row in rows))

train_pairs, test_pairs = train_test_split(
    pair_ids,
    test_size=0.2,
    random_state=42
)

train_pairs = set(train_pairs)
test_pairs = set(test_pairs)

train_rows = [row for row in rows if row["pair_id"] in train_pairs]
test_rows = [row for row in rows if row["pair_id"] in test_pairs]

print("Total samples:", len(rows))
print("Training samples:", len(train_rows))
print("Testing samples:", len(test_rows))

print("Training pairs:", len(train_pairs))
print("Testing pairs:", len(test_pairs))

features = [
    "loop_count",
    "condition_count",
    "call_count",
    "subscript_access_count",
    "max_nesting_depth",
    "max_loop_nesting",
    "is_recursive",
    "function_length"
]

X_train = [
    [int(row[feature]) for feature in features]
    for row in train_rows
]

y_train = [
    int(row["label"])
    for row in train_rows
]

X_test = [
    [int(row[feature]) for feature in features]
    for row in test_rows
]

y_test = [
    int(row["label"])
    for row in test_rows
]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))