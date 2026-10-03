"""Independent TF-IDF retrieval for the Without-QA module."""
import math
from collections import Counter

def cosine_rank(query, target):
    """Whole-file token cosine, no path boosts or graph/QA candidates."""
    df = Counter(t for d in target.values() for t in d["tokens"])
    idf = {t: math.log((1 + len(target)) / (1 + n)) + 1 for t, n in df.items()}
    q = {t: (1 + math.log(n)) * idf.get(t, math.log(1 + len(target)) + 1)
         for t, n in query.items()}
    qnorm = math.sqrt(sum(x*x for x in q.values())) or 1
    scored = []
    for path, d in target.items():
        weights = {t: (1 + math.log(n)) * idf[t] for t, n in d["tokens"].items()}
        norm = math.sqrt(sum(x*x for x in weights.values())) or 1
        score = sum(q.get(t, 0) * x for t, x in weights.items()) / (qnorm * norm)
        if score > 0:
            scored.append({"path": path, "score": min(1, score)})
    return sorted(scored, key=lambda x: (-x["score"], x["path"]))
