import json
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data():
    import json

    with open("dataset/codecomplex/python_data.jsonl", encoding="utf-8") as file:
        return [json.loads(line) for line in file]
        
input_path = "dataset/codecomplex/python_data.jsonl"

with open(input_path, encoding="utf-8") as file:
    rows = [json.loads(line) for line in file]

problems = sorted(set(row["problem"] for row in rows))

train_problems, test_problems = train_test_split(
    problems,
    test_size=0.2,
    random_state=42
)

train_problems = set(train_problems)
test_problems = set(test_problems)

train_rows = [row for row in rows if row["problem"] in train_problems]
test_rows = [row for row in rows if row["problem"] in test_problems]

print("Total samples:", len(rows))
print("Training samples:", len(train_rows))
print("Testing samples:", len(test_rows))
print("Training problems:", len(train_problems))
print("Testing problems:", len(test_problems))

print("\nTraining classes:")
print(Counter(row["complexity"] for row in train_rows))

print("\nTesting classes:")
print(Counter(row["complexity"] for row in test_rows))