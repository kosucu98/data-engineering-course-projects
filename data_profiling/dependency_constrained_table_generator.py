#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import pandas as pd


# ----------------- Checks -----------------
def approx_fd_errors_A_to_H(df: pd.DataFrame) -> int:
    # sum_A ( |group(A)| - max_h freq in that group )
    total = 0
    for _, g in df.groupby("A", sort=False):
        counts = g["H"].value_counts()
        total += len(g) - int(counts.max())
    return int(total)


def minimal_uccs(df: pd.DataFrame) -> list[frozenset[str]]:
    cols = list(df.columns)

    def is_unique(subset: tuple[str, ...]) -> bool:
        return not df.duplicated(subset=list(subset)).any()

    unique_sets = []
    for r in range(1, len(cols) + 1):
        for subset in combinations(cols, r):
            if is_unique(subset):
                unique_sets.append(frozenset(subset))

    mins = []
    for s in unique_sets:
        if not any(t < s for t in unique_sets):
            mins.append(s)

    return sorted(mins, key=lambda x: (len(x), tuple(sorted(x))))


def single_attribute_inds(df: pd.DataFrame) -> list[tuple[str, str]]:
    cols = list(df.columns)
    inds = []
    sets = {c: set(df[c].tolist()) for c in cols}
    for x in cols:
        for y in cols:
            if x == y:
                continue
            if sets[x].issubset(sets[y]):
                inds.append((x, y))
    return sorted(inds)


def validate(df: pd.DataFrame) -> None:
    # UCCs
    assert df["G"].is_unique, "UCC {G} violated."
    assert not df.duplicated(subset=["F", "H"]).any(), "UCC {F,H} violated."

    # Given FDs
    assert (df.groupby("B")["C"].nunique() <= 1).all(), "FD B->C violated."
    assert (df.groupby("B")["D"].nunique() <= 1).all(), "FD B->D violated."
    assert (df.groupby("E")["A"].nunique() <= 1).all(), "FD E->A violated."

    # Given IND
    assert set(df["A"]).issubset(set(df["D"])), "IND A ⊂ D violated."

    # Approx FD
    errors = approx_fd_errors_A_to_H(df)
    assert errors == 5, f"Approx FD A->H has {errors} errors, expected 5."

    # Exact minimal UCCs
    mins = minimal_uccs(df)
    expected = [frozenset({"G"}), frozenset({"F", "H"})]
    assert mins == expected, f"Unexpected minimal UCCs found: {mins}"

    # Exact single-attribute INDs: ONLY A ⊂ D
    inds = single_attribute_inds(df)
    expected_inds = [("A", "D")]
    assert inds == expected_inds, f"Unexpected IND(s) found: {inds}"

    print("Validation passed:")
    print(" - minimal UCCs:", mins)
    print(" - A->H errors:", errors)
    print(" - single-attr INDs:", inds)


# ----------------- Generator -----------------
def generate_table(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    # D universe (strings). A must be a proper subset of D.
    D_vals = [f"v{idx:02d}" for idx in range(60)]   # v00..v59
    A_vals = D_vals[:45]                            # v00..v44  (A ⊂ D)

    # H values in their own namespace (prevents H ⊂ A/C/D etc.)
    H_vals = [f"h{idx:02d}" for idx in range(15)]   # h00..h14

    def a_to_h(a: str) -> str:
        # map v00..v44 -> h00..h14 (non-injective)
        a_idx = int(a[1:])  # from "vXX"
        return H_vals[a_idx % 15]

    # B domain and FD B -> C,D
    n_B = 120
    B_vals = [f"B{idx:03d}" for idx in range(n_B)]
    C_vals = [f"c{idx:02d}" for idx in range(30)]   # c00..c29 (separate namespace)

    def b_to_d(i: int) -> str:
        # D is from v00..v59
        return D_vals[i % 60]

    def b_to_c(i: int) -> str:
        # MUST break D -> C: if i and i+60 share the same D,
        # they should not necessarily share C.
        return C_vals[(i * 7 + (i // 60) * 11) % 30]

    B_to_D = {b: b_to_d(i) for i, b in enumerate(B_vals)}
    B_to_C = {b: b_to_c(i) for i, b in enumerate(B_vals)}

    # E domain and FD E -> A (separate namespace)
    n_E = 180
    E_vals = [f"E{idx:03d}" for idx in range(n_E)]

    assigned_As = (A_vals * ((n_E // len(A_vals)) + 1))[:n_E]
    rng.shuffle(assigned_As)
    E_to_A = {e: a for e, a in zip(E_vals, assigned_As)}

    # Allocate F so (F,H) is unique, but F alone is NOT unique.
    # Use F values in their own namespace to avoid accidental INDs.
    used_f_by_h: dict[str, set[str]] = defaultdict(set)
    next_f_by_h: dict[str, int] = defaultdict(int)

    def alloc_f(h: str) -> str:
        k = next_f_by_h[h]
        # allow same F across different H, but not within same H
        f = f"f{k:05d}"
        while f in used_f_by_h[h]:
            k += 1
            f = f"f{k:05d}"
        used_f_by_h[h].add(f)
        next_f_by_h[h] = k + 1
        return f

    rows: list[dict[str, str]] = []

    # Reserve 10 rows for 5 A->H errors (paired rows)
    normal_n = n - 10
    if normal_n <= 0:
        raise ValueError("n must be >= 11")

    # Ensure IND A ⊂ D holds robustly:
    # Force D to contain all A values v00..v44 at least once.
    for idx in range(45):
        B = B_vals[idx]
        D = B_to_D[B]        # will be v00..v44
        C = B_to_C[B]
        E = rng.choice(E_vals)
        A = E_to_A[E]        # some v00..v44
        H = a_to_h(A)
        F = alloc_f(H)
        G = f"G{len(rows):04d}"
        rows.append({"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "H": H})

    # Block 1: same B, varying E (break unintended B->E etc., without breaking B->C,D)
    B0 = B_vals[0]
    C0, D0 = B_to_C[B0], B_to_D[B0]
    for _ in range(15):
        E = rng.choice(E_vals)
        A = E_to_A[E]
        H = a_to_h(A)
        F = alloc_f(H)
        G = f"G{len(rows):04d}"
        rows.append({"A": A, "B": B0, "C": C0, "D": D0, "E": E, "F": F, "G": G, "H": H})

    # Block 2: same E, varying B (break unintended E->B/C/D)
    E0 = E_vals[0]
    A0 = E_to_A[E0]
    H0 = a_to_h(A0)
    for j in range(20):
        B = B_vals[(j + 1) % len(B_vals)]
        C, D = B_to_C[B], B_to_D[B]
        F = alloc_f(H0)
        G = f"G{len(rows):04d}"
        rows.append({"A": A0, "B": B, "C": C, "D": D, "E": E0, "F": F, "G": G, "H": H0})

    # Fill remaining normal rows
    while len(rows) < normal_n:
        B = rng.choice(B_vals)
        C = B_to_C[B]
        D = B_to_D[B]
        E = rng.choice(E_vals)
        A = E_to_A[E]
        H = a_to_h(A)
        F = alloc_f(H)
        G = f"G{len(rows):04d}"
        rows.append({"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "H": H})

    # Inject exactly 5 A->H errors and prevent accidental extra UCCs.
    rng.shuffle(rows)
    chosen: list[dict[str, str]] = []
    seen_A: set[str] = set()
    for r in rows:
        if r["A"] not in seen_A:
            chosen.append(r)
            seen_A.add(r["A"])
        if len(chosen) == 5:
            break
    if len(chosen) < 5:
        raise RuntimeError("Could not find 5 distinct A values to inject errors.")

    for k, base in enumerate(chosen):
        A, B, C, D, E = base["A"], base["B"], base["C"], base["D"], base["E"]
        H_correct = a_to_h(A)

        # choose a wrong H different from the correct one
        correct_idx = H_vals.index(H_correct)
        H_wrong = H_vals[(correct_idx + 1) % len(H_vals)]

        # same F for both rows in the pair -> creates duplicates in combinations such as (E,F)
        F = f"fx{k:03d}"

        # Ensure (F,H) remains unique
        if F in used_f_by_h[H_correct] or F in used_f_by_h[H_wrong]:
            raise RuntimeError("Reserved F collided unexpectedly. Change prefix fx or k-range.")
        used_f_by_h[H_correct].add(F)
        used_f_by_h[H_wrong].add(F)

        G1 = f"G{len(rows):04d}"
        rows.append({"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G1, "H": H_correct})

        G2 = f"G{len(rows):04d}"
        rows.append({"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G2, "H": H_wrong})

    df = pd.DataFrame(rows, columns=["A", "B", "C", "D", "E", "F", "G", "H"])
    if len(df) != n:
        raise RuntimeError(f"Internal error: produced {len(df)} rows, expected {n}.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and validate a synthetic table with predefined data dependencies."
    )
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path(__file__).with_name("generated_dependency_table.csv"),
        help="Path where the generated CSV file will be written.",
    )
    args = parser.parse_args()

    df = generate_table(n=args.rows, seed=args.seed)
    validate(df)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)

    print(f"\nWrote {len(df)} rows to {args.output_file}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()