from datasets import load_dataset

def main():
    # Choose one subset (start with MuSiQue)
    subset_name = "musique"

    print(f"Loading LongBench subset: {subset_name} ...")

    # Load dataset
    dataset = load_dataset("THUDM/LongBench", subset_name)

    # Inspect available splits
    print("\nAvailable splits:", dataset.keys())

    # Most LongBench subsets use "test" split
    split = "test"
    data = dataset[split]

    print(f"\nNumber of samples in '{split}' split:", len(data))

    # Take first sample
    sample = data[0]

    print("\n--- Sample Keys ---")
    print(sample.keys())

    print("\n--- Question ---")
    print(sample.get("input"))

    print("\n--- Context ---")
    context = sample.get("context", "No context field found")
    print(context[:1000])  # Print first 1000 characters only 

    print("\n--- Ground Truth Answer ---")
    print(sample.get("answers"))



if __name__ == "__main__":
    main()
