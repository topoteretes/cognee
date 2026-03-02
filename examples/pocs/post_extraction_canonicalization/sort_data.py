import re

# Usage: python sort_matches.py input.txt
path = "results.txt"

pattern = re.compile(
    r"node=(.*?),\s*closest_match=\[array\(\['(.*?)'\],\s*dtype=object\),\s*array\(\[([0-9]*\.?[0-9]+)\]\)\]"
)

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line.strip())
        if m:
            node, match, score = m.groups()
            rows.append((node.strip(), match.strip(), float(score), line.strip()))

# sort by score descending
rows.sort(key=lambda x: x[2], reverse=True)

for node, match, score, _ in rows:
    print(f"node={node}, closest={match}, score={score:.4f}")
