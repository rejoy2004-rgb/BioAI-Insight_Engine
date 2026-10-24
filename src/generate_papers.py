"""Generate a clearly-labelled SYNTHETIC document set for load/latency testing only.

The original version of this script produced 510 fake "papers" with real journal
names, invented authors and invented results, and the app indexed them as if
they were literature. That is misleading, so it was replaced:

  * output goes to data/synthetic_corpus.json (NOT indexed by default),
  * every record says it is synthetic, with placeholder authors/journal,
  * records carry no fabricated findings or numbers.

To index the output anyway (e.g. to benchmark retrieval at scale), add an
entry with `"synthetic": true` and inline `"text"` to papers/sources.json.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json"

TOPICS = ["breast cancer", "lung cancer", "colorectal cancer", "prostate cancer", "melanoma", "leukemia"]
METHODS = ["logistic regression", "random forests", "convolutional networks", "gene expression profiling", "image analysis"]
AIMS = ["screening", "survival analysis", "biomarker discovery", "histopathology", "treatment response"]


def generate(n: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        topic, method, aim = rng.choice(TOPICS), rng.choice(METHODS), rng.choice(AIMS)
        docs.append({
            "id": f"synthetic-{i + 1:04d}",
            "title": f"[SYNTHETIC] Placeholder document {i + 1}: {method} and {aim} in {topic}",
            "authors": "Synthetic Author",
            "journal": "Synthetic test corpus (not a real publication)",
            "year": None,
            "synthetic": True,
            "text": (f"This placeholder text mentions {topic}, {method} and {aim} so that lexical retrieval can be "
                     f"exercised. It contains no findings and must not be cited. Document number {i + 1}."),
        })
    return docs


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(generate(), indent=2), encoding="utf-8")
    print(f"Wrote {OUT} (synthetic, not indexed by default)")
