from datasets import load_dataset

dataset = load_dataset("OSS-forge/PyResBugs", split="train")

for i in range(20):
    row = dataset[i]

    print("\n--- ROW", i, "---")
    print("Fixed Method:", row["Fixed_Method"])