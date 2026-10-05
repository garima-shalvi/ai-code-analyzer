import csv

with open("dataset/ml_dataset.csv", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("Number of samples:", len(rows))
print("Columns:", reader.fieldnames)

print("\nFirst 5 rows:")

for row in rows[:5]:
    print(row)