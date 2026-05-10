import re
from collections import defaultdict

INPUT_FILE = r"C:\Users\oguzh\Desktop\Homework Augsburg\Algorithm_Data_Engineering\universities.txt"
OUT_TSV    = r"C:\Users\oguzh\Desktop\Homework Augsburg\Algorithm_Data_Engineering\university_matches.tsv"

STOP = {
    "university","universities","institute","college","school","faculty","dept","department",
    "of","and","for","the","a","an","in","to","at","campus","main","branch","deemed",
    "technology","technological","engineering","sciences","science","applied","research",
    "international","national","state","city","islamic","autonomous","affiliated","affliated"
}

def normalize(s: str) -> str:
    s = re.sub(r"https?://\S+", "", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def tokens(norm: str):
    return {t for t in norm.split() if t and t not in STOP and len(t) > 1}

def jaccard(a, b) -> float:
    if not a and not b: return 1.0
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0

def lev_dist(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def lev_sim(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - lev_dist(a, b) / m

# --- read input ---
raw = []
with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if s:
            raw.append(s)

norms = [normalize(s) for s in raw]
toks  = [tokens(n) for n in norms]

# --- union-find ---
parent = list(range(len(raw)))
rank = [0]*len(raw)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb: return
    if rank[ra] < rank[rb]: ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]: rank[ra] += 1

# --- blocking (avoid full O(n^2)) ---
buckets = defaultdict(list)
for i, n in enumerate(norms):
    ts = sorted(toks[i])
    key1 = (n[:6] if n else "")
    key2 = " ".join(ts[:3])

    # FIX: use [] not ()
    buckets[("p", key1)].append(i)
    if key2:
        buckets[("t", key2)].append(i)

# --- compare within buckets ---
seen_pairs = set()
for idxs in buckets.values():
    if len(idxs) < 2:
        continue
    idxs = sorted(set(idxs))
    for a in range(len(idxs)):
        i = idxs[a]
        for b in range(a+1, len(idxs)):
            j = idxs[b]
            if (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))

            jac = jaccard(toks[i], toks[j])
            if jac >= 0.80:
                union(i, j)
                continue

            if len(toks[i] & toks[j]) >= 2:
                if lev_sim(norms[i], norms[j]) >= 0.92:
                    union(i, j)

# --- collect clusters (only duplicates) ---
clusters = defaultdict(list)
for i in range(len(raw)):
    clusters[find(i)].append(raw[i])

groups = [v for v in clusters.values() if len(v) >= 2]
groups.sort(key=lambda g: (-len(g), g[0].lower()))

# --- write TSV: ID, Match1, Match2, ...
with open(OUT_TSV, "w", encoding="utf-8", newline="") as out:
    for gid, g in enumerate(groups, 1):
        out.write(str(gid) + "\t" + "\t".join(g) + "\n")

print(f"Wrote {len(groups)} matched groups to: {OUT_TSV}")
