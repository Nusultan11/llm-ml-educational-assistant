import re


class TokenOverlapReranker:
    def rerank(self, query: str, chunk_records: list, top_k: int) -> list:
        scored = []

        for original_rank, record in enumerate(chunk_records):
            score = self._score(query, record)
            scored.append((score, original_rank, record))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:top_k]]

    def _score(self, query: str, record) -> float:
        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0

        text_terms = set(self._tokenize(record.text))
        title_terms = set(self._tokenize(f"{record.title} {record.section or ''}"))
        query_term_set = set(query_terms)

        text_hits = len(query_term_set & text_terms)
        title_hits = len(query_term_set & title_terms)

        coverage_score = text_hits / len(query_term_set)
        title_score = title_hits / len(query_term_set)

        normalized_query = " ".join(query_terms)
        normalized_text = " ".join(self._tokenize(record.text))
        normalized_title = " ".join(self._tokenize(f"{record.title} {record.section or ''}"))

        exact_query_bonus = 0.25 if normalized_query and normalized_query in normalized_text else 0.0
        title_bonus = 0.15 if normalized_query and normalized_query in normalized_title else 0.0

        return coverage_score + (0.35 * title_score) + exact_query_bonus + title_bonus

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())
