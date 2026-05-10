# Data Engineering Course Projects

This repository contains selected coding projects completed for the **Algorithms & Data Engineering** course at the **University of Augsburg**.

The projects cover topics in:

- data profiling
- entity matching
- schema matching

## Projects

### Data Profiling

The `data_profiling/` folder contains a script that generates and validates a synthetic table with predefined data dependencies, including functional dependencies, an approximate functional dependency, unique column combinations, and an inclusion dependency.

### Entity Matching

The `entity_matching/` folder contains a script for identifying duplicate or near-duplicate university names using text normalization, blocking, Jaccard similarity, Levenshtein similarity, and union-find clustering.

### Schema Matching

The `schema_matching/` folder contains a schema matching pipeline that combines BERT-based semantic similarity, Jaccard similarity, weighted score aggregation, and one-to-one matching with the Hungarian algorithm.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data_profiling/
│   ├── README.md
│   ├── dependency_constrained_table_generator.py
│   └── generated_dependency_table.csv
├── entity_matching/
│   ├── README.md
│   └── university_name_matching.py
└── schema_matching/
    ├── README.md
    └── schema_matching_pipeline.py
```

## Notes

- These projects were developed as part of coursework and are published with the course instructor's permission.
- The external datasets used for the entity matching and schema matching projects are not included in this repository. Links and usage instructions are provided in the corresponding folder-level README files.
- Each project folder contains a short README with further details and example usage instructions.
