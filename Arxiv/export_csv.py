import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset


def main():
    custom_cache_dir = "/data/zwt/huggingface/datasets"
    print("Loading dataset...")
    # Load the dataset
    dataset = load_dataset("leminda-ai/s2orc_small", cache_dir=custom_cache_dir)

    # Columns to exclude
    exclude_columns = [
        "id",
        "entities",
        "pdfUrls",
        "s2PdfUrl",
        "inCitations",
        "outCitations",
        "s2Url",
    ]

    # Function to process and save a dataset split
    def process_and_save(ds, split_name):
        # Identify columns to remove that actually exist in the dataset
        cols_to_remove = [col for col in exclude_columns if col in ds.column_names]

        if cols_to_remove:
            print(f"Removing columns from {split_name}: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)

        # Keep only rows whose abstract has at least 1500 characters
        print(f"Filtering {split_name} for abstract length > 1500...")
        ds = ds.filter(
            lambda example: example.get("paperAbstract") is not None
            and len(example["paperAbstract"]) > 1000
        )

        # Randomly select 100,000 rows if dataset is larger than that
        if len(ds) > 100000:
            print(
                f"Shuffling and selecting 100,000 rows from {split_name} (total: {len(ds)})..."
            )
            ds = ds.shuffle(seed=42).select(range(100000))
        else:
            print(f"Dataset {split_name} has {len(ds)} rows (<= 100,000), keeping all.")

        output_path = os.path.join("Arxiv", f"s2orc_small_{split_name}.csv")
        print(f"Saving {split_name} to {output_path}...")
        ds.to_csv(output_path)
        print(f"Finished saving {output_path}")

    # Handle DatasetDict (multiple splits) or single Dataset
    if hasattr(dataset, "keys") and callable(getattr(dataset, "keys")):
        for split in dataset.keys():
            process_and_save(dataset[split], split)
    else:
        process_and_save(dataset, "data")


if __name__ == "__main__":
    main()
