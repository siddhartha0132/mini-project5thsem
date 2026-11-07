"""Step 4a: extractive summarization with TextRank.

Each sentence is a node in a graph. Edges are weighted by the cosine
similarity of the sentences' TF-IDF vectors. PageRank then scores every
sentence; the score is boosted for sentences that contain medical entities,
abnormal lab values or come from important sections (diagnosis, impression).
The best sentences are returned in their original order.
"""

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# How much each report section matters in a summary.
SECTION_WEIGHT = {
    "Diagnosis": 1.6,
    "Impression": 1.6,
    "Chief Complaint": 1.3,
    "Hospital Course": 1.3,
    "Findings": 1.2,
    "Treatment": 1.1,
    "Procedures": 1.3,
    "Medications": 0.8,  # also listed separately in the entity panel
    "Investigations": 1.0,
    "Advice": 1.0,
    "Follow-up": 1.0,
    "History": 0.9,
    "Past History": 0.8,
    "Examination": 0.8,
    "General": 0.7,
}

DAMPING = 0.85


def textrank_scores(sentences):
    """Plain TextRank: PageRank over the sentence-similarity graph.

    Returns (scores, similarity matrix)."""
    n = len(sentences)
    if n == 1:
        return np.array([1.0]), np.zeros((1, 1))
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = tfidf.fit_transform(sentences)
    except ValueError:  # only stop words
        return np.ones(n) / n, np.zeros((n, n))
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    graph = nx.from_numpy_array(similarity)
    ranks = nx.pagerank(graph, alpha=DAMPING, weight="weight", max_iter=200)
    return np.array([ranks[i] for i in range(n)]), similarity


def summarize(sentence_items, entities, lab_results, ratio=0.3, max_sentences=8, min_sentences=2):
    """Pick the most important sentences.

    sentence_items: list of {"text", "section"} from preprocessing.
    entities: entity dicts (used to boost informative sentences).
    lab_results: lab readings (sentences with abnormal values are boosted).
    """
    texts = [s["text"] for s in sentence_items]
    if not texts:
        return {"summary": "", "sentences": []}
    if len(texts) <= min_sentences:
        return {"summary": " ".join(texts), "sentences": [{"text": t, "score": 1.0} for t in texts]}

    scores, similarity = textrank_scores(texts)
    scores = scores / scores.max()

    entity_terms = {e["norm"] for e in entities if not e["negated"]}
    abnormal_sources = [r["source"].lower() for r in lab_results if r["status"] != "NORMAL"]
    final = []
    for i, item in enumerate(sentence_items):
        lower = item["text"].lower()
        boost = SECTION_WEIGHT.get(item["section"], 1.0)
        boost *= 1 + 0.15 * min(3, sum(term in lower for term in entity_terms))
        if any(src in lower for src in abnormal_sources):
            boost *= 1.3
        if len(item["text"].split()) < 4:
            boost *= 0.6
        final.append(scores[i] * boost)
    final = np.array(final)

    k = max(min_sentences, min(max_sentences, int(round(len(texts) * ratio))))
    chosen = []
    for i in np.argsort(-final):
        # Skip near-duplicates of an already chosen sentence.
        if any(similarity[i][j] > 0.6 for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen) == k:
            break
    chosen.sort()
    picked = [{"text": texts[i], "score": round(float(final[i]), 3), "section": sentence_items[i]["section"]}
              for i in chosen]
    return {"summary": " ".join(p["text"] for p in picked), "sentences": picked}
