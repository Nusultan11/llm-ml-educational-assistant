from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RetrievalQuality:
    sufficient: bool
    reason: str
    best_score: float
    average_score: float
    best_coverage: float
    strong_results: int
    total_results: int

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "best_score": round(self.best_score, 6),
            "average_score": round(self.average_score, 6),
            "best_coverage": round(self.best_coverage, 6),
            "strong_results": self.strong_results,
            "total_results": self.total_results,
        }


class RetrievalQualityGate:
    def __init__(
        self,
        min_score: float,
        min_coverage: float,
        min_strong_results: int,
    ):
        self.min_score = min_score
        self.min_coverage = min_coverage
        self.min_strong_results = min_strong_results

    def assess(self, query: str, chunk_records: list) -> RetrievalQuality:
        if not chunk_records:
            return RetrievalQuality(
                sufficient=False,
                reason="No contexts were retrieved.",
                best_score=0.0,
                average_score=0.0,
                best_coverage=0.0,
                strong_results=0,
                total_results=0,
            )

        scores = [self._score(query, record) for record in chunk_records]
        coverages = [coverage for _, coverage in scores]
        weighted_scores = [score for score, _ in scores]

        best_score = max(weighted_scores)
        average_score = sum(weighted_scores) / len(weighted_scores)
        best_coverage = max(coverages)
        strong_results = sum(1 for score in weighted_scores if score >= self.min_score)

        sufficient = (
            best_score >= self.min_score
            and best_coverage >= self.min_coverage
            and strong_results >= self.min_strong_results
        )

        if sufficient:
            reason = "Retrieved context passed the quality gate."
        elif best_coverage < self.min_coverage:
            reason = "Retrieved context only partially overlaps with the question."
        elif best_score < self.min_score:
            reason = "Retrieved context is too weak to support a reliable answer."
        else:
            reason = "Not enough strong chunks were retrieved."

        return RetrievalQuality(
            sufficient=sufficient,
            reason=reason,
            best_score=best_score,
            average_score=average_score,
            best_coverage=best_coverage,
            strong_results=strong_results,
            total_results=len(chunk_records),
        )

    def _score(self, query: str, record) -> tuple[float, float]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0, 0.0

        query_set = set(query_terms)
        text_terms = set(self._tokenize(record.text))
        title_terms = set(self._tokenize(f"{record.title} {record.section or ''}"))

        text_hits = len(query_set & text_terms)
        title_hits = len(query_set & title_terms)

        coverage = text_hits / len(query_set)
        title_coverage = title_hits / len(query_set)
        score = coverage + (0.35 * title_coverage)
        return score, coverage

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())
