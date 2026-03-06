import math
import re
from collections import Counter


class KeywordIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_tokens = []
        self.doc_freqs = []
        self.idf = {}
        self.avg_doc_len = 0.0

    def build(self, documents: list[str]):
        self.documents = documents
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        num_docs = len(documents)
        if num_docs == 0:
            self.idf = {}
            self.avg_doc_len = 0.0
            return

        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / num_docs

        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))

        self.idf = {
            term: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[int]:
        if not self.documents:
            return []

        query_terms = self._tokenize(query)
        scored = []

        for idx, term_freq in enumerate(self.doc_freqs):
            score = self._score_document(term_freq, len(self.doc_tokens[idx]), query_terms)
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [idx for score, idx in scored[:top_k] if score > 0]

    def _score_document(self, term_freq: Counter, doc_len: int, query_terms: list[str]) -> float:
        score = 0.0

        for term in query_terms:
            if term not in term_freq:
                continue

            tf = term_freq[term]
            idf = self.idf.get(term, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-9))
            )
            score += idf * (numerator / max(denominator, 1e-9))

        return score

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\\w+", text.lower())
