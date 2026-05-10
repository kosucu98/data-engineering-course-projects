#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer


class SchemaMatchingPipeline:
    def __init__(self, bert_weight=0.6, jaccard_weight=0.4, threshold=0.3):
        print("Initializing BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()

        self.bert_weight = bert_weight
        self.jaccard_weight = jaccard_weight
        self.threshold = threshold
        print("Pipeline ready!\n")

    # MATCHERS

    def bert_matcher(self, schema1: pd.DataFrame, schema2: pd.DataFrame) -> np.ndarray:
        """BERT-based semantic matcher."""

        def encode_column(col_name, col_data):
            text = f"{col_name}: {', '.join(map(str, col_data.dropna().head(5).tolist()))}"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[0, 0, :].numpy()

        emb1 = np.array([encode_column(col, schema1[col]) for col in schema1.columns])
        emb2 = np.array([encode_column(col, schema2[col]) for col in schema2.columns])
        return cosine_similarity(emb1, emb2)

    def jaccard_matcher(self, schema1: pd.DataFrame, schema2: pd.DataFrame) -> np.ndarray:
        """Jaccard similarity based on unique column values."""
        n_cols1 = len(schema1.columns)
        n_cols2 = len(schema2.columns)
        similarity = np.zeros((n_cols1, n_cols2))

        for i, col1 in enumerate(schema1.columns):
            set1 = {str(v).lower() for v in schema1[col1].dropna().unique()}
            for j, col2 in enumerate(schema2.columns):
                set2 = {str(v).lower() for v in schema2[col2].dropna().unique()}
                if len(set1) == 0 and len(set2) == 0:
                    similarity[i, j] = 1.0
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity[i, j] = intersection / union if union > 0 else 0.0

        return similarity

    # COMBINER

    def combiner(self, bert_sim: np.ndarray, jaccard_sim: np.ndarray) -> np.ndarray:
        """Weighted linear combination of matcher scores."""
        return self.bert_weight * bert_sim + self.jaccard_weight * jaccard_sim

    # SELECTOR

    def selector(
        self,
        similarity_matrix: np.ndarray,
        schema1_cols: List[str],
        schema2_cols: List[str],
    ) -> List[Tuple[str, str, float]]:
        """Hungarian algorithm for one-to-one matches."""
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        matches = []
        for i, j in zip(row_ind, col_ind):
            sim = similarity_matrix[i, j]
            if sim >= self.threshold:
                matches.append((schema1_cols[i], schema2_cols[j], sim))

        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    # MAIN PIPELINE

    def match(self, schema1: pd.DataFrame, schema2: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Run the full schema matching pipeline."""
        print(f"Matching schemas: {len(schema1.columns)} cols vs {len(schema2.columns)} cols")

        print("  Running BERT matcher...")
        bert_sim = self.bert_matcher(schema1, schema2)

        print("  Running Jaccard matcher...")
        jaccard_sim = self.jaccard_matcher(schema1, schema2)

        print("  Combining similarities...")
        combined_sim = self.combiner(bert_sim, jaccard_sim)

        print("  Selecting one-to-one matches...")
        matches = self.selector(combined_sim, list(schema1.columns), list(schema2.columns))

        print(f"\n  Found {len(matches)} matches:")
        for col1, col2, sim in matches:
            print(f"    {col1:30s} <-> {col2:30s} ({sim:.3f})")

        return matches


# EVALUATION

def load_ground_truth_json(json_path: Path) -> set[tuple[str, str]]:
    """Load ground-truth matches from a JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = set()
    if "matches" in data:
        for match in data["matches"]:
            source_col = match.get("source_column")
            target_col = match.get("target_column")
            if source_col and target_col:
                ground_truth.add((source_col, target_col))

    return ground_truth


def evaluate(matches: List[Tuple[str, str, float]], ground_truth: set) -> Dict[str, float]:
    """Evaluate predicted matches against ground truth."""
    predicted = {(col1, col2) for col1, col2, _ in matches}

    tp = len(predicted.intersection(ground_truth))
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a schema matching pipeline on selected Valentine benchmark datasets."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        required=True,
        help="Path to the local Valentine-datasets directory.",
    )
    parser.add_argument("--bert_weight", type=float, default=0.6)
    parser.add_argument("--jaccard_weight", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 80)
    print("SCHEMA MATCHING PIPELINE")
    print("=" * 80)
    print()

    datasets_to_test = [
        {
            "path": "ChEMBL/Unionable/assays_horizontal_0_ec_ev",
            "file1": "assays_horizontal_0_ec_ev_source.csv",
            "file2": "assays_horizontal_0_ec_ev_target.csv",
            "ground_truth": "assays_horizontal_0_ec_ev_mapping.json",
        },
        {
            "path": "ChEMBL/Unionable/assays_horizontal_0_ac2_av",
            "file1": "assays_horizontal_0_ac2_av_source.csv",
            "file2": "assays_horizontal_0_ac2_av_target.csv",
            "ground_truth": "assays_horizontal_0_ac2_av_mapping.json",
        },
        {
            "path": "Magellan/amazon_google_exp",
            "file1": "amazon_google_exp_source.csv",
            "file2": "amazon_google_exp_target.csv",
            "ground_truth": "amazon_google_exp_mapping.json",
        },
    ]

    pipeline = SchemaMatchingPipeline(
        bert_weight=args.bert_weight,
        jaccard_weight=args.jaccard_weight,
        threshold=args.threshold,
    )

    all_results = []

    for dataset_config in datasets_to_test:
        dataset_name = dataset_config["path"]
        file1_name = dataset_config["file1"]
        file2_name = dataset_config["file2"]
        gt_filename = dataset_config.get("ground_truth")

        print("\n" + "=" * 80)
        print(f"TESTING: {dataset_name}")
        print("=" * 80)

        try:
            dataset_path = args.base_path / dataset_name

            schema1 = pd.read_csv(dataset_path / file1_name)
            schema2 = pd.read_csv(dataset_path / file2_name)

            print(f"Schema 1: {schema1.shape[0]} rows, {schema1.shape[1]} columns")
            print(f"Schema 2: {schema2.shape[0]} rows, {schema2.shape[1]} columns")
            print()

            matches = pipeline.match(schema1, schema2)

            if gt_filename:
                gt_path = dataset_path / gt_filename

                if os.path.exists(gt_path):
                    print(f"\n✓ Loading ground truth: {gt_filename}")
                    ground_truth = load_ground_truth_json(gt_path)
                    print(f"  Ground truth size: {len(ground_truth)} matches")

                    metrics = evaluate(matches, ground_truth)

                    print("\n" + "-" * 80)
                    print("EVALUATION RESULTS:")
                    print("-" * 80)
                    print(f"Precision:        {metrics['precision']:.3f}")
                    print(f"Recall:           {metrics['recall']:.3f}")
                    print(f"F1 Score:         {metrics['f1_score']:.3f}")
                    print(f"True Positives:   {metrics['true_positives']}")
                    print(f"False Positives:  {metrics['false_positives']}")
                    print(f"False Negatives:  {metrics['false_negatives']}")

                    if metrics["false_positives"] > 0 or metrics["false_negatives"] > 0:
                        predicted = {(col1, col2) for col1, col2, _ in matches}

                        wrong_matches = predicted - ground_truth
                        if wrong_matches:
                            print(f"\n Incorrect matches ({len(wrong_matches)}):")
                            for col1, col2 in list(wrong_matches)[:5]:
                                print(f"     {col1} <-> {col2}")

                        missed_matches = ground_truth - predicted
                        if missed_matches:
                            print(f"\n Missed matches ({len(missed_matches)}):")
                            for col1, col2 in list(missed_matches)[:5]:
                                print(f"     {col1} <-> {col2}")

                    all_results.append(
                        {
                            "dataset": dataset_name,
                            "matches": len(matches),
                            **metrics,
                        }
                    )
                else:
                    print(f"\n Ground truth file not found: {gt_path}")
                    all_results.append(
                        {
                            "dataset": dataset_name,
                            "matches": len(matches),
                            "precision": None,
                            "recall": None,
                            "f1_score": None,
                        }
                    )
            else:
                print("\n no ground truth specified")
                all_results.append(
                    {
                        "dataset": dataset_name,
                        "matches": len(matches),
                        "precision": None,
                        "recall": None,
                        "f1_score": None,
                    }
                )

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_results:
        print(f"\n{'Dataset':<40} {'Matches':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-" * 90)

        for result in all_results:
            dataset_short = result["dataset"].split("/")[-1]
            precision_str = f"{result['precision']:.3f}" if result["precision"] is not None else "N/A"
            recall_str = f"{result['recall']:.3f}" if result["recall"] is not None else "N/A"
            f1_str = f"{result['f1_score']:.3f}" if result["f1_score"] is not None else "N/A"

            print(
                f"{dataset_short:<40} "
                f"{result['matches']:<10} "
                f"{precision_str:<12} "
                f"{recall_str:<12} "
                f"{f1_str:<12}"
            )

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()