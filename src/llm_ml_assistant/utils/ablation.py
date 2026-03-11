from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalVariant:
    retrieval_mode: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def parse_csv_ints(value: str, field_name: str, min_value: int = 1) -> list[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{field_name} cannot be empty")

    values: list[int] = []
    seen: set[int] = set()

    for part in parts:
        try:
            parsed = int(part)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains non-integer value: {part}") from exc

        if parsed < min_value:
            raise ValueError(f"{field_name} values must be >= {min_value}: {parsed}")

        if parsed not in seen:
            seen.add(parsed)
            values.append(parsed)

    return values


def parse_csv_strings(value: str, field_name: str) -> list[str]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{field_name} cannot be empty")

    values: list[str] = []
    seen: set[str] = set()

    for part in parts:
        if part not in seen:
            seen.add(part)
            values.append(part)

    return values


def safe_run_label(label: str, max_len: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip())
    return (cleaned[:max_len] or "run").strip("_") or "run"


def generate_retrieval_variants(
    chunk_sizes: list[int],
    chunk_overlaps: list[int],
    top_ks: list[int],
    retrieval_modes: list[str],
) -> list[RetrievalVariant]:
    allowed_modes = {"vector", "hybrid"}
    variants: list[RetrievalVariant] = []

    for mode in retrieval_modes:
        if mode not in allowed_modes:
            raise ValueError(f"retrieval_mode must be one of {sorted(allowed_modes)}: {mode}")

        for chunk_size in chunk_sizes:
            for overlap in chunk_overlaps:
                # Skip invalid chunking profiles.
                if overlap >= chunk_size:
                    continue
                for top_k in top_ks:
                    variants.append(
                        RetrievalVariant(
                            retrieval_mode=mode,
                            chunk_size=chunk_size,
                            chunk_overlap=overlap,
                            top_k=top_k,
                        )
                    )

    if not variants:
        raise ValueError("No valid variants were generated. Check chunk_size/chunk_overlap ranges.")

    return variants
