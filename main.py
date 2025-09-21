
# main.py : entry point for Financial RAG

import json
from pathlib import Path

# === copy your helper functions here ===
# acquire_filings, build_corpus, build_all_chunks, build_index_for_doc, ask, etc.

def run_pipeline():
    # Acquire + parse filings
    saved_files = acquire_filings()
    DOCS = build_corpus()
    ALL_CHUNKS = build_all_chunks()

    # Build embeddings/index (your INDEXES code from notebook)

    # Queries
    queries = [
        "What was Microsoft's total revenue in 2023?",
        "How did NVIDIA’s data center revenue grow from 2022 to 2023?",
        "Which company had the highest operating margin in 2023?",
        "What percentage of Google's revenue came from cloud in 2023?",
        "Compare AI investments mentioned by all three companies in their 2024 10-Ks"
    ]
    outputs = [ask(q) for q in queries]

    out_path = Path("sec_10k/sample_outputs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print("Results saved to", out_path)
    print(json.dumps(outputs, indent=2)[:1000])

if __name__ == "__main__":
    run_pipeline()
