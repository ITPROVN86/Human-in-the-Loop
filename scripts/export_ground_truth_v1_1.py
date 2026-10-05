from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

input_file = (
    ROOT
    / "data"
    / "v1_1"
    / "FPT_STDD_2026_Final_Human_Validation_and_GT_v1_1.xlsx"
)

output_dir = ROOT / "data" / "v1_1" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "ground_truth_v1_1.csv"
excluded_file = output_dir / "unresolved_pairs_v1_1.csv"

df = pd.read_excel(
    input_file,
    sheet_name="Ground_Truth_v1_1",
)

required_columns = [
    "annotation_id",
    "topic_a_id",
    "topic_b_id",
    "title_a_en",
    "title_b_en",
    "ground_truth_v1_1",
]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

valid_labels = ["Duplicate", "Non-duplicate"]

resolved = df[
    df["ground_truth_v1_1"].isin(valid_labels)
].copy()

unresolved = df[
    ~df["ground_truth_v1_1"].isin(valid_labels)
].copy()

resolved["label"] = (
    resolved["ground_truth_v1_1"]
    .map(
        {
            "Duplicate": 1,
            "Non-duplicate": 0,
        }
    )
    .astype(int)
)

resolved["dataset_version"] = "FPT-STDD-2026-v1.1"

resolved.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig",
)

unresolved.to_csv(
    excluded_file,
    index=False,
    encoding="utf-8-sig",
)

print("Export completed")
print(f"Resolved pairs   : {len(resolved)}")
print(f"Duplicate        : {(resolved['label'] == 1).sum()}")
print(f"Non-duplicate    : {(resolved['label'] == 0).sum()}")
print(f"Unresolved       : {len(unresolved)}")
print(f"Output           : {output_file}")
print(f"Excluded output  : {excluded_file}")