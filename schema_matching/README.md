# Schema Matching

This folder contains a schema matching pipeline that identifies correspondences between columns from two datasets.

The pipeline combines:

- BERT-based semantic similarity using column names and sample values
- Jaccard similarity based on overlapping column values
- weighted score combination
- one-to-one match selection with the Hungarian algorithm
- evaluation against ground-truth mappings using precision, recall, and F1-score

## Files

- `schema_matching_pipeline.py` — implements the full schema matching pipeline and evaluates it on selected benchmark datasets

## Dataset

The script was developed using selected datasets from the **Valentine Datasets** benchmark for schema matching.

The dataset is available here:

https://zenodo.org/records/5084605#.YOgWHBMzY-Q

The datasets are not included in this repository. Download them separately and provide the local dataset path when running the script.

## Usage

After downloading and extracting the Valentine Datasets, run:

```bash
python schema_matching_pipeline.py \
  --base_path /path/to/Valentine-datasets
```

Optional parameters:

```bash
python schema_matching_pipeline.py \
  --base_path /path/to/Valentine-datasets \
  --bert_weight 0.6 \
  --jaccard_weight 0.4 \
  --threshold 0.3
```

## Evaluated Dataset Pairs

The script evaluates the pipeline on the following dataset pairs:

- `ChEMBL/Unionable/assays_horizontal_0_ec_ev`
- `ChEMBL/Unionable/assays_horizontal_0_ac2_av`
- `Magellan/amazon_google_exp`
