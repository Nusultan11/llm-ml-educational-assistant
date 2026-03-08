import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


TOPIC_KEYWORDS = {
    "ml",
    "machine learning",
    "deep learning",
    "neural",
    "transformer",
    "attention",
    "llm",
    "rag",
    "embedding",
    "faiss",
    "python",
    "pytorch",
    "tensorflow",
    "huggingface",
    "fine-tune",
    "finetune",
}


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_topic_match(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in TOPIC_KEYWORDS)


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_rag_item(store: list[dict[str, Any]], source: str, item_id: str, title: str, text: str):
    title = normalize_text(title)
    text = normalize_text(text)
    if not text:
        return

    merged = f"{title}\n\n{text}" if title else text
    if not is_topic_match(merged):
        return

    store.append(
        {
            "id": item_id,
            "source": source,
            "title": title,
            "text": text,
            "tags": ["ml_assistant"],
        }
    )


def add_sft_item(store: list[dict[str, Any]], source: str, item_id: str, instruction: str, response: str):
    instruction = normalize_text(instruction)
    response = normalize_text(response)
    if not instruction or not response:
        return

    if not is_topic_match(instruction + " " + response):
        return

    store.append(
        {
            "id": item_id,
            "source": source,
            "instruction": instruction,
            "response": response,
        }
    )


def load_openassistant(max_samples: int, rag_rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]]):
    if load_dataset is None:
        raise RuntimeError("datasets library is not installed. pip install datasets")

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    count = 0
    for row in ds:
        text = normalize_text(str(row.get("text", "")))
        if not text:
            continue

        item_id = str(row.get("message_id", f"oasst-{count}"))
        role = str(row.get("role", "")).lower()

        add_rag_item(rag_rows, "openassistant", item_id, "", text)

        if role == "assistant":
            continue

        if count + 1 < len(ds):
            next_row = ds[count + 1]
            if str(next_row.get("role", "")).lower() == "assistant":
                add_sft_item(
                    sft_rows,
                    "openassistant",
                    item_id,
                    text,
                    normalize_text(str(next_row.get("text", ""))),
                )

        count += 1
        if count >= max_samples:
            break


def load_dolly(max_samples: int, rag_rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]]):
    if load_dataset is None:
        raise RuntimeError("datasets library is not installed. pip install datasets")

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    for i, row in enumerate(ds):
        instruction = normalize_text(str(row.get("instruction", "")))
        context = normalize_text(str(row.get("context", "")))
        response = normalize_text(str(row.get("response", "")))

        full_instruction = instruction if not context else f"{instruction}\nContext: {context}"
        item_id = str(row.get("id", f"dolly-{i}"))

        add_sft_item(sft_rows, "dolly", item_id, full_instruction, response)
        add_rag_item(rag_rows, "dolly", item_id, instruction, response)

        if i + 1 >= max_samples:
            break


def load_local_stackoverflow(path: Path, max_samples: int, rag_rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]]):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            q = normalize_text(str(row.get("question", "")))
            a = normalize_text(str(row.get("answer", "")))
            title = normalize_text(str(row.get("title", "")))
            item_id = str(row.get("id", f"stackoverflow-{i}"))

            add_sft_item(sft_rows, "stackoverflow", item_id, q, a)
            add_rag_item(rag_rows, "stackoverflow", item_id, title or q, a)

            if i + 1 >= max_samples:
                break


def load_local_arxiv(path: Path, max_samples: int, rag_rows: list[dict[str, Any]]):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            title = normalize_text(str(row.get("title", "")))
            abstract = normalize_text(str(row.get("abstract", "")))
            item_id = str(row.get("id", f"arxiv-{i}"))

            add_rag_item(rag_rows, "arxiv", item_id, title, abstract)

            if i + 1 >= max_samples:
                break


def main():
    parser = argparse.ArgumentParser(description="Prepare unified RAG and SFT datasets for ML assistant.")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")
    parser.add_argument("--max-openassistant", type=int, default=30000)
    parser.add_argument("--max-dolly", type=int, default=15000)
    parser.add_argument("--max-stackoverflow", type=int, default=30000)
    parser.add_argument("--max-arxiv", type=int, default=30000)
    parser.add_argument("--no-openassistant", action="store_true")
    parser.add_argument("--no-dolly", action="store_true")
    parser.add_argument("--stackoverflow-path", default="data/raw/stackoverflow.jsonl")
    parser.add_argument("--arxiv-path", default="data/raw/arxiv.jsonl")

    args = parser.parse_args()

    rag_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []

    if not args.no_openassistant:
        print("Loading OpenAssistant...")
        load_openassistant(args.max_openassistant, rag_rows, sft_rows)

    if not args.no_dolly:
        print("Loading Dolly...")
        load_dolly(args.max_dolly, rag_rows, sft_rows)

    print("Loading local StackOverflow JSONL (if exists)...")
    load_local_stackoverflow(Path(args.stackoverflow_path), args.max_stackoverflow, rag_rows, sft_rows)

    print("Loading local ArXiv JSONL (if exists)...")
    load_local_arxiv(Path(args.arxiv_path), args.max_arxiv, rag_rows)

    out_dir = Path(args.out_dir)
    rag_path = out_dir / "rag_corpus.jsonl"
    sft_path = out_dir / "sft_instructions.jsonl"

    write_jsonl(rag_path, rag_rows)
    write_jsonl(sft_path, sft_rows)

    summary = {
        "rag_rows": len(rag_rows),
        "sft_rows": len(sft_rows),
        "rag_path": str(rag_path),
        "sft_path": str(sft_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
