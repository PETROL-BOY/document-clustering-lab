import argparse
import csv
import sys
import time
from pathlib import Path

# Ensure app package is importable
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.pipeline import ingest_file, chunk_documents, run_k_sweep_pipeline
from app.config import settings

def main():
    parser = argparse.ArgumentParser(description="Repeatable DWM K-Sweep Experiment Runner")
    parser.add_argument("--dataset", type=str, default="data/sample_documents.csv", help="Path to corpus file")
    parser.add_argument("--method", type=str, default="tfidf", choices=["tfidf", "gemini"], help="Representation method")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum K value")
    parser.add_argument("--k-max", type=int, default=6, help="Maximum K value")
    parser.add_argument("--output-csv", type=str, default="experiments/k_sweep_results.csv", help="Path to output CSV")

    args = parser.parse_args()

    dataset_path = backend_dir / args.dataset
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found.")
        sys.exit(1)

    print(f"=== Starting DWM K-Sweep Experiment ===")
    print(f"Dataset: {args.dataset}")
    print(f"Representation: {args.method.upper()}")
    print(f"K range: [{args.k_min}, {args.k_max}]")
    if args.method == "gemini":
        print(f"Embedding model: {settings.gemini_embed_model}")

    with open(dataset_path, "rb") as f:
        content = f.read()

    docs = ingest_file(dataset_path.name, content)
    chunks = chunk_documents(docs)
    print(f"Ingested {len(docs)} documents -> {len(chunks)} chunks.")

    t0 = time.time()
    results = run_k_sweep_pipeline(
        chunks=chunks,
        k_min=args.k_min,
        k_max=args.k_max,
        method=args.method
    )
    elapsed = time.time() - t0

    output_path = backend_dir / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "inertia", "silhouette", "sizes", "method", "dataset", "elapsed_s"])
        for r in results:
            writer.writerow([
                r["k"],
                r["inertia"],
                r["silhouette"],
                str(r["sizes"]),
                args.method,
                dataset_path.name,
                round(elapsed, 3)
            ])

    print(f"\nExperiment complete in {elapsed:.2f}s! Results saved to {output_path}")
    print("\n--- Summary Table ---")
    print(f"{'K':<5} | {'Inertia (WCSS)':<15} | {'Cosine Silhouette':<18} | {'Cluster Sizes'}")
    print("-" * 65)
    for r in results:
        sil_str = f"{r['silhouette']:.4f}" if r['silhouette'] is not None else "N/A"
        print(f"{r['k']:<5} | {r['inertia']:<15.4f} | {sil_str:<18} | {r['sizes']}")

if __name__ == "__main__":
    main()
