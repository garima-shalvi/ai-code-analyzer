from datasets import load_dataset

dataset = load_dataset("OSS-forge/PyResBugs", split="train")

print("Number of rows:", len(dataset))
print("Columns:")
print(dataset.column_names)

for i in range(3):
    row = dataset[i]

    print("\n--- ROW", i, "---")
    print("Bug Type:", row["Bug_Type"])
    print("Fixed Method:", row["Fixed_Method"])
    print("Faulty Code:")
    print(row["Faulty Code"])
    print("Fault Free Code:")
    print(row["Fault Free Code"])