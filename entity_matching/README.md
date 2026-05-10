# Entity Matching

This folder contains a script for identifying duplicate or near-duplicate university names.

The matching pipeline uses:

- text normalization
- blocking
- Jaccard similarity
- Levenshtein similarity
- union-find clustering

## Files

- `university_name_matching.py` — identifies duplicate or near-duplicate university names

## Dataset

The script was developed using a university-name dataset provided through the course materials.  
The dataset is available here:

https://mediastore.rz.uni-augsburg.de/file/bc5384326624f3d03d71bdc26c514586/6a00af6a/MsN0d11eKS/universities.txt

The dataset is not included in this repository. Download it separately and provide its path when running the script.

## Usage

```bash
python university_name_matching.py \
  --input_file /path/to/universities.txt \
  --output_file university_matches.tsv
```
