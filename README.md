<div align="center">

# RobustRAG

**An Empirical Study of Corpus Poisoning and Query Noise in Retrieval-Augmented Question Answering**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22153313.svg)](https://zenodo.org/records/22153313)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20preprint-orange.svg)](https://zenodo.org/records/22153313)

**Maryam Zaheer** · Department of Computer Science · University of Engineering and Technology (UET), Lahore
📄 [Read the preprint on Zenodo](https://zenodo.org/records/22153313) &nbsp;|&nbsp; 📖 [Citation](#citation)

</div>

---

## Overview

RobustRAG is a controlled, reproducible evaluation framework for measuring the robustness of Retrieval-Augmented Generation (RAG) pipelines under two independent failure modes: **corpus poisoning** and **query-side embedding noise**. It measures how each perturbation affects both retrieval quality and downstream answer quality, and evaluates whether a common lightweight defense — near-duplicate suppression — can recover the resulting degradation.

> This is an evaluation study, not a new attack or defense algorithm. Its main contribution is a mechanistically-explained negative result: a standard near-duplicate suppression defense, implemented the way most practitioners would build it on a first attempt, measurably fails at the one thing it is designed to do — and we identify exactly why.

## Table of Contents

- [Key Finding](#key-finding)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Key Finding

Query-side embedding noise — not corpus poisoning — is the dominant threat to answer quality at the settings tested. More significantly, the evaluated defense (near-duplicate suppression) **does not recover the retrieval loss it targets and instead makes Recall@5 measurably worse** (up to −4.6% relative to the poisoned condition).

This traces to a structural property of the algorithm: it builds its kept-candidate list greedily in descending score order and always retains the top-ranked candidate unconditionally. As a result, it is **structurally incapable** of changing what a downstream reader sees, regardless of how the similarity threshold is tuned — and this limitation generalizes to any defense that filters a fixed-size candidate list top-down without reconsidering the top result.

Full derivation in the preprint, Section V-C.

---

## Research Questions

| # | Question |
|---|----------|
| 1 | How sensitive is RAG retrieval to different forms of corpus poisoning? |
| 2 | How severely does query-side embedding noise affect retrieval and answer quality? |
| 3 | Can a lightweight near-duplicate suppression strategy recover performance after corpus poisoning? |

Corpus-level and query-level perturbations are evaluated independently, on the same retriever and corpus, so their relative severity can be directly compared rather than assumed.

---

## Methodology

### Dataset

**SQuAD 1.1 validation split** — used deliberately rather than the training split, since the extractive reader (`distilbert-base-cased-distilled-squad`) was fine-tuned on SQuAD's training data. Evaluating on the validation split avoids the reader answering from memorized associations rather than the retrieved passage.

| | |
|---|---|
| Articles | 48 |
| Unique passages (post-dedup) | 2,067 |
| Answerable questions | 10,570 |

### Retrieval System

- **Embedding model:** `BAAI/bge-small-en-v1.5` via `sentence-transformers`
- **Asymmetric encoding:** queries receive an instruction prefix (`"Represent this sentence for searching relevant passages:"`); passages do not
- **Index:** FAISS `IndexFlatIP` (exact inner-product search over L2-normalized embeddings)

### Perturbations Evaluated

**Corpus poisoning** (5% injection rate, 103 documents), all derived from real corpus content:

| Strategy | Description |
|---|---|
| Near-duplicate | Verbatim copy of a real passage |
| Contradictory | Copy with the correct answer replaced by an incorrect one drawn from elsewhere in the corpus; falls back to a plain duplicate if substring substitution fails |
| Irrelevant | Word-shuffled passage — shares vocabulary, no coherent meaning |

**Query-side noise:** additive Gaussian noise (σ = 0.1) applied to query embeddings, then renormalized, evaluated independently of any corpus change.

### Defense: Near-Duplicate Suppression

Given a candidate pool of 20 (4× the final top-*k* of 5), candidates are kept in descending score order only if their embedding similarity to every already-kept candidate is below a cosine threshold of **0.97**; otherwise they are skipped and backfilled from lower-ranked candidates.

### Metrics

| Retrieval | Answer Quality |
|---|---|
| Recall@5 | Exact Match |
| Precision@5 | BLEU |
| Mean Reciprocal Rank (MRR) | ROUGE-L |

Answer extraction uses only the **top-1** retrieved passage, so answer quality remains directly attributable to top-ranked retrieval quality.

---

## Results

All metrics computed over the full set of 10,570 queries. Fixed random seed = 42.

| Condition | Recall@5 | Precision@5 | MRR | EM | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| **Clean (baseline)** | 0.887 | 0.177 | 0.763 | 0.559 | 0.259 | 0.631 |
| Near-duplicate, poisoned | 0.884 | 0.177 | 0.744 | 0.559 | 0.259 | 0.631 |
| Near-duplicate, defended | 0.843 | 0.169 | 0.726 | 0.559 | 0.259 | 0.631 |
| Contradictory, poisoned | 0.883 | 0.177 | 0.750 | 0.558 | 0.258 | 0.629 |
| Contradictory, defended | 0.873 | 0.175 | 0.750 | 0.558 | 0.258 | 0.629 |
| Irrelevant, poisoned | 0.886 | 0.177 | 0.761 | 0.557 | 0.258 | 0.628 |
| Irrelevant, defended | 0.886 | 0.177 | 0.761 | 0.557 | 0.258 | 0.628 |
| **Query noise (σ = 0.1)** | 0.603 | 0.121 | 0.443 | 0.290 | 0.136 | 0.342 |

**Query noise dominates.** Relative to clean: Recall@5 −32%, MRR −42%, Exact Match −48%. Every metric degrades in the same direction and by a comparable magnitude, consistent with noise acting on the query representation directly.

**Poisoning at 5% is modest but mechanistically consistent.** Near-duplicate poisoning leaves EM, BLEU, and ROUGE-L numerically identical to baseline — expected, since a near-duplicate poisoned document is textually identical to the one it copies. Contradictory poisoning is the only strategy that alters document content, and correspondingly the only one that moves answer-quality metrics.

**The defense does not recover retrieval loss — and cannot affect answer quality by construction.** Because the suppression algorithm always keeps the top-ranked candidate unconditionally, the passage handed to the reader is identical before and after defense for every query. What the defense *does* change — ranks 2 through 5 — moved net negative in every poisoning condition tested.

---

## Repository Structure

```text
RobustRAG/
├── run.py
├── config/
│   └── default.yaml
├── data/
│   └── README.md
├── results/
│   ├── logs/
│   └── figures/
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── retrieval/
│   ├── attacks/
│   ├── defense/
│   └── evaluation/
└── tests/
```

| Component | Purpose |
|---|---|
| `data/` | Dataset loading and corpus construction |
| `retrieval/` | Sentence embeddings, FAISS indexing, retrieval |
| `attacks/` | Corpus poisoning and query-noise generation |
| `defense/` | Near-duplicate suppression |
| `evaluation/` | Retrieval metrics, extractive QA, reporting |
| `run.py` | Unified experiment entry point |

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Download SQuAD 1.1 and set its path in config/default.yaml
```

**Default configuration:**

| Parameter | Value |
|---|---|
| Corpus | SQuAD 1.1 validation split, 2,067 passages |
| Retrieval depth | Top-5 (candidate pool of 20 for defense) |
| Poisoning rate | 5% (103 documents) |
| Query noise level | σ = 0.1 |
| Defense similarity threshold | 0.97 |
| Random seed | 42 |

**Run the full experiment suite:**

```bash
python run.py --experiment baseline

python run.py --experiment poisoning --strategy near_duplicate
python run.py --experiment poisoning --strategy contradictory
python run.py --experiment poisoning --strategy irrelevant

python run.py --experiment noise

python run.py --experiment defense --strategy near_duplicate
python run.py --experiment defense --strategy contradictory
python run.py --experiment defense --strategy irrelevant

python run.py --experiment report
```

Use `--skip-answers` to evaluate retrieval only, without the extractive reader.

> **Reproducibility note:** minor non-determinism (4th-decimal-place differences in Recall@5 for the contradictory condition across identical runs) was observed, likely from floating-point summation order under multi-threaded CPU execution in FAISS/PyTorch. This does not affect any conclusion but is reported for transparency.

---

## Limitations

- All poisoning experiments use a single injection rate (5%); a dose-response sweep is needed to know how the defense's negative retrieval effect scales.
- Point estimates are reported without a paired significance test (e.g., McNemar's on per-query outcomes); the current pipeline logs only aggregate metrics.
- Poisoning strategies are heuristic and corpus-derived, not gradient-optimized adversarial attacks (cf. Zhong et al. 2023; Zou et al. 2024 — PoisonedRAG). Stronger attacks would likely produce larger effects, though the defense's structural failure mode is independent of how the poisoned document was constructed.
- The extractive reader is limited to a contiguous span of the retrieved passage and cannot correct, synthesize, or hedge.
- Recall@k, Precision@k, and MRR assume exactly one relevant passage per query, matching this SQuAD-derived setup but not corpora with multiple relevant passages.
- Query noise is evaluated at a single level (σ = 0.1); a sweep is needed for a full robustness curve.
- The defense targets embedding-level similarity specifically and is not a general-purpose poisoning defense.

---

## Future Work

1. Redesign the defense to reconsider the top-ranked candidate rather than protect it by construction.
2. Sweep poisoning rates (5%, 10%, 20%, 30%) to test scaling of the defense's negative retrieval effect.
3. Add per-query outcome logging to support paired significance testing.
4. Benchmark against stronger defenses — cross-encoder reranking, semantic clustering, evidence-consistency filtering.
5. Extend to a generative reader and evaluate answer faithfulness, not just extractive correctness.
6. Evaluate additional datasets and embedding models to test generalization.

---

## Citation

If you use this repository, please cite the accompanying preprint:

```bibtex
@misc{zaheer2025robustrag,
  author       = {Zaheer, Maryam},
  title        = {RobustRAG: An Empirical Study of Corpus Poisoning and Query Noise
                  in Retrieval-Augmented Question Answering},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22153313},
  url          = {https://zenodo.org/records/22153313}
}
```

---

## License

Released under the [MIT License](LICENSE).
