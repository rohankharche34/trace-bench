# Trace-Bench (OpenMemoryBench)

A lightweight benchmarking framework for evaluating and comparing memory/retrieval systems used in LLM applications. Trace-Bench measures how well different vector-based memory architectures handle semantically noisy queries and reports accuracy across four difficulty tiers, along with vector-space geometry and hubness diagnostics.

## Motivation

LLM agents often rely on external memory systems (vector databases, embedding-based retrieval) to store and recall facts. In production, user queries are rarely clean — they contain typos, missing words, and character deletions. Trace-Bench quantifies how well a given memory system degrades under such conditions, helping practitioners choose the right retrieval backend for their use case.

## Features

- **Multi-tier corruption pipeline** — Queries are perturbed at three difficulty levels (Easy, Medium, Hard) using composable noise primitives (typos, word dropout, character deletion).
- **Pluggable memory systems** — Subclass `BaseMemory` to add your own backend (FAISS, Chroma, Pinecone, etc.).
- **Accuracy & robustness metrics** — Per-tier accuracy, robustness ratio (hard/clean), and timing (index + search).
- **Embedding-space diagnostics** — Anisotropy metrics (eigenvalue analysis) and hubness analysis (k-NN neighbor frequency skew) to detect collapsed or degenerate embedding spaces.
- **Synthetic dataset generation** — Optional Gemini-powered factoid QA generation via the `instructor` library.

## Project Structure

```
trace-bench/
├── run_bench.py                 # Main entry point
├── requirements.txt             # Python dependencies
├── .env                         # Gemini API key (for dataset generation)
├── .gitignore
├── LICENSE
│
├── benchmarks/
│   ├── runner.py                # EvaluationRunner — orchestrates full benchmark
│   └── noise.py                 # BenchmarkCorruptor — tiered noise pipelines
│
├── corruptions/
│   ├── typo.py                  # Adjacent character swap
│   ├── dropout.py               # Random word dropout
│   └── delete.py                # Single character deletion
│
├── datasets/
│   ├── database.py              # Gemini-powered synthetic QA generator
│   └── synthetic_data.csv       # Pre-generated 100 factoid pairs
│
├── metrics/
│   ├── anisotropy.py            # Embedding-space geometry (covariance, eigenvalues)
│   └── hubness.py               # k-NN hubness / universal-neighbor detection
│
└── systems/
    ├── base.py                  # Abstract BaseMemory interface
    ├── cosine_memory.py         # Pure NumPy cosine-similarity memory
    └── faiss_memory.py          # FAISS IndexFlatIP (inner product) memory
```

## Quick Start

```bash
# Clone & setup
cd trace-bench
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the benchmark
python run_bench.py
```

The benchmark loads 100 pre-generated factoid pairs, indexes them into each memory system, then queries every row with both the clean query and three corrupted variants. Results are printed as a formatted table.

### Additional dependencies

If you want to regenerate the synthetic dataset:

```bash
pip install instructor google-genai pydantic python-dotenv scipy
```

Then set your Gemini API key in `.env`:

```
GEMINI_API_KEY=your_key_here
```

And run:

```bash
python datasets/database.py
```

## Adding a New Memory System

1. Create a new file in `systems/`, e.g. `systems/chroma_memory.py`.
2. Subclass `BaseMemory` and implement:
   - `store(memories: list[str])` — embed and index all strings.
   - `retrieve(query: str) -> str | None` — return the closest memory string.
   - `get_all_embeddings() -> np.ndarray` — return the full embedding matrix.
3. Register it in `run_bench.py`:

```python
from systems.chroma_memory import ChromaMemory

systems = [CosineMemory(), FaissMemory(), ChromaMemory()]
```

## Adding a New Corruption Type

1. Create a function in `corruptions/` that takes a string and returns a string.
2. Import and compose it in `benchmarks/noise.py` inside `BenchmarkCorruptor.corrupt()`.

## Metrics Explained

### Accuracy Tiers

| Tier    | Corruption Applied                                                   |
|---------|----------------------------------------------------------------------|
| Clean   | No corruption                                                        |
| Easy    | 50% typo OR 50% word dropout (p=0.1)                                 |
| Medium  | Character deletion + typo                                            |
| Hard    | Word dropout (p=0.35) + typo + character deletion                    |

### Robustness

`robustness = hard_acc / clean_acc` — the fraction of clean accuracy retained under the hardest noise tier.

### Anisotropy

Measures how "stretched" the embedding space is via eigenvalue analysis of the covariance matrix. High anisotropy ratios indicate embeddings are concentrated along a few dimensions, which can harm discrimination.

### Hubness

Hubness skewness measures the tendency for a few points to be "universal neighbors" of many others — a known curse-of-dimensionality phenomenon. Positive skewness indicates hubness.

## Included Systems

| System         | Backend          | Similarity          | Notes                            |
|----------------|------------------|---------------------|----------------------------------|
| `CosineMemory` | NumPy            | Cosine similarity   | Reference implementation         |
| `FaissMemory`  | FAISS            | Inner product (L2 normalized) | Scales to larger datasets |

Both use `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings).

## License

MIT — see [LICENSE](LICENSE).

## Author

Rohan Kharche
