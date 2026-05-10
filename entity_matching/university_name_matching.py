#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from pathlib import Path


STOP = {
    "university", "universities", "institute", "college", "school", "faculty",
    "dept", "department", "of", "and", "for", "the", "a", "an", "in", "to",
    "at", "campus", "main", "branch", "deemed", "technology", "technological",
    "engineering", "sciences", "science", "applied", "research", "international",
    "national", "state", "city", "islamic", "autonomous", "affiliated", "affliated"
}


def normalize(s: str) -> str:
    s = re.sub(r"https?://\S+", "", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def tokens(norm: str) -> set[str]:
    return {t for t in norm.split() if t and t not in STOP and len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def lev_dist(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur

    return prev[-1]


def lev_sim(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    return 1.0 if max_len == 0 else 1.0 - lev_dist(a, b) / max_len


def load_records(input_file: Path) -> list[str]:
    records = []
    with input_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(s)
    return records


def match_universities(raw: list[str]) -> list[list[str]]:
    norms = [normalize(s) for s in raw]
    toks = [tokens(n) for n in norms]

    # Union-find
    parent = list(range(len(raw)))
    rank = [0] * len(raw)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Blocking to avoid full O(n^2) comparison
    buckets = defaultdict(list)
    for i, norm in enumerate(norms):
        sorted_tokens = sorted(toks[i])
        key1 = norm[:6] if norm else ""
        key2 = " ".join(sorted_tokens[:3])

        buckets[("p", key1)].append(i)
        if key2:
            buckets[("t", key2)].append(i)

    # Compare within blocks
    seen_pairs = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue

        idxs = sorted(set(idxs))
        for a in range(len(idxs)):
            i = idxs[a]
            for b in range(a + 1, len(idxs)):
                j = idxs[b]

                if (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))

                jac = jaccard(toks[i], toks[j])
                if jac >= 0.80:
                    union(i, j)
                    continue

                if len(toks[i] & toks[j]) >= 2 and lev_sim(norms[i], norms[j]) >= 0.92:
                    union(i, j)

    # Collect duplicate clusters
    clusters = defaultdict(list)
    for i in range(len(raw)):
        clusters[find(i)].append(raw[i])

    groups = [group for group in clusters.values() if len(group) >= 2]
    groups.sort(key=lambda group: (-len(group), group[0].lower()))

    return groups


def write_matches(groups: list[list[str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as out:
        for gid, group in enumerate(groups, 1):
            out.write(str(gid) + "\t" + "\t".join(group) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify duplicate or near-duplicate university names using blocking and similarity matching."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the input text file containing one university name per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path(__file__).with_name("university_matches.tsv"),
        help="Path where the matched groups will be written.",
    )
    args = parser.parse_args()

    raw = load_records(args.input_file)
    groups = match_universities(raw)
    write_matches(groups, args.output_file)

    print(f"Wrote {len(groups)} matched groups to: {args.output_file}")


if __name__ == "__main__":
    main()