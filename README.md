# RobustRAG: Evaluating Retrieval Robustness Against Corpus Poisoning

A small evaluation framework for studying how corpus poisoning and query-side
noise affect retrieval quality and downstream answer quality in a
Retrieval-Augmented Generation (RAG) pipeline, and how much of that
degradation a lightweight defense can recover.

This is a research-oriented evaluation project, not a novel method. The goal
is to demonstrate a clean, reproducible experimental setup for studying RAG
robustness, not to propose a new algorithm.

## Motivation

RAG systems are only as trustworthy as the corpus they retrieve from. If an
attacker can insert documents into that corpus — near-duplicates of real
content, subtly false variants of real facts, or noisy irrelevant text — the
retriever may surface poisoned content just as readily as genuine content,
and a downstream reader has no way to tell the difference. Separately, the
query side can also degrade: a noisy encoder, a corrupted embedding, or an
imperfect query rewrite can push a legitimate question away from its correct
answer even with a perfectly clean corpus.

This project isolates and measures both failure modes, and tests whether a
simple, explainable defense can recover some of the lost accuracy.

## Project Overview

The pipeline:

1. Load a subset of SQuAD and pool its paragraphs into a retrieval corpus
2. Embed the corpus with a sentence embedding model and index it with FAISS
3. Measure baseline retrieval and answer quality
4. Inject poisoned documents into the corpus (three strategies) and re-measure
5. Perturb query embeddings with gaussian noise and re-measure, independently
   of poisoning
6. Apply a lightweight near-duplicate suppression defense on the poisoned
   index and measure how much of the degradation is recovered

Every stage is driven by `run.py` and a single YAML config, so any experiment
can be reproduced with one command.

## Repository Structure

```
RobustRAG/
├── run.py                     # single entry point (baseline/poisoning/noise/defense/report)
├── config/default.yaml        # all experiment parameters
├── data/                       # SQuAD json goes here (not committed, see data/README.md)
├── results/
│   ├── logs/                  # per-experiment JSON metric dumps
│   └── figures/                # generated figures
├── src/
│   ├── data/loader.py          # SQuAD parsing + corpus subsampling
│   ├── retrieval/               # embeddings, FAISS index, retriever
│   ├── attacks/                # corpus poisoning + query noise
│   ├── defense/                # near-duplicate suppression
│   └── evaluation/             # metrics, reader, evaluator, plots
└── tests/                      # sanity tests for the non-ML-dependent logic
```

## Methodology

**Embedding model.** Retrieval uses `BAAI/bge-small-en-v1.5` via
`sentence-transformers`, not CLIP. CLIP's text encoder is trained for
image-text alignment, not text-text semantic similarity, and is not a strong
fit for pure-text retrieval. BGE is a small, CPU-friendly model trained
specifically for this task.

**Corpus.** A subset of SQuAD's paragraphs (default: 2,500), deduplicated and
pooled into a single retrieval corpus. Each answerable question is treated as
a query with a known gold passage, which makes Recall@k, Precision@k, and MRR
directly computable without extra annotation.

**Poisoning strategies**, all derived from real corpus content and appended
to the corpus (originals are never removed, so degradation can be attributed
specifically to the injected documents):
- **near-duplicate** — verbatim copies of real passages
- **contradictory** — a real passage with its true answer substituted for a
  wrong one, so the poisoned document is topically identical to a real
  passage but asserts a false fact
- **irrelevant** — a word-shuffled version of a real passage: similar
  vocabulary, no coherent meaning

**Retrieval noise** is gaussian perturbation applied directly to query
embeddings (then renormalized), kept independent of corpus poisoning so the
two failure modes can be measured separately before being combined.

**Defense.** A plain similarity threshold on retrieval scores would not catch
near-duplicate or contradictory poisoning, since both are constructed to
score just as high as the real passage they were derived from. Instead, the
defense retrieves a larger candidate pool and greedily keeps the
highest-scoring candidates while skipping any whose embedding is
near-identical (cosine similarity ≥ 0.97 by default) to one already kept,
backfilling from the pool until `k` results remain.

**Answer quality.** A small extractive QA model
(`distilbert-base-cased-distilled-squad`) extracts an answer span from the
top-retrieved passage. This is deliberately extractive rather than
generative — it isolates retrieval's effect on the answer without adding an
LLM's own error modes into the picture.

## Experimental Setup

All parameters live in `config/default.yaml`. Defaults: 2,500-passage corpus,
top-5 retrieval, 5% poisoning rate, noise level 0.1, similarity threshold
0.97 for the defense.

```bash
pip install -r requirements.txt

# download SQuAD (train or dev) from https://rajpurkar.github.io/SQuAD-explorer/
# and place it at the path in config/default.yaml (data.squad_path)

python run.py --experiment baseline
python run.py --experiment poisoning --strategy near_duplicate
python run.py --experiment poisoning --strategy contradictory
python run.py --experiment poisoning --strategy irrelevant
python run.py --experiment noise
python run.py --experiment defense --strategy near_duplicate
python run.py --experiment report   # renders figures from whatever logs exist
```

Add `--skip-answers` to any experiment to skip the extractive-reader step and
get faster retrieval-only numbers while iterating.

Every run is seeded (`data.seed` in the config) for reproducibility — the
corpus subsample, poisoning injection, and noise perturbation are all
deterministic given the same seed.

## Results

Results are not committed to this repository, since they depend on which
SQuAD subset is downloaded and are regenerated by running the commands above.
After running the full sweep, `python run.py --experiment report` produces:

- `retrieval_accuracy.png` / `answer_accuracy.png` — baseline performance
- `poisoning_impact.png` — Recall@k under each poisoning strategy vs. clean
- `defense_recovery_<strategy>.png` — clean vs. poisoned vs. defended Recall@k
- `results_summary.csv` — every condition's full metric set in one tidy table

Expected qualitative pattern, based on how each strategy is constructed:
near-duplicate and contradictory poisoning should degrade retrieval the most,
since both are built as embedding-close copies of real passages that
compete directly with the genuine document. The irrelevant strategy should
degrade retrieval less, since shuffled text sits further from the query in
embedding space. The defense specifically targets embedding-near-duplicate
content, so it should recover a meaningful fraction of the near-duplicate and
contradictory degradation, but is not expected to help much against the
irrelevant strategy, which isn't a near-duplicate attack in the first place.

## Limitations

- Evaluation uses SQuAD's dev/validation split (`dev-v1.1.json`), not train. This
  matters specifically because the extractive reader
  (`distilbert-base-cased-distilled-squad`) was fine-tuned on SQuAD's training
  split — evaluating it on train data would let it potentially answer from
  memorized question-answer pairs rather than from the retrieved context,
  which would undermine the retrieval-to-answer causal story this project is
  built around. The retriever (BGE) has no such leakage concern, since it was
  never fine-tuned on SQuAD specifically.
- Poisoning strategies are heuristic constructions, not adversarially
  optimized attacks against the retriever (e.g. no gradient-based embedding
  attacks).
- The contradictory strategy falls back to plain duplication when the true
  answer string doesn't appear verbatim in its source passage; the effective
  contradiction rate should be checked against the fallback rate on the real
  dataset.
- Recall@k, Precision@k, and MRR assume exactly one relevant passage per
  query, which holds for this SQuAD-derived setup but doesn't generalize
  directly to multi-relevant-passage retrieval settings.
- The reader only sees the top-1 retrieved passage, not a multi-passage
  context, so answer quality is tightly coupled to top-1 retrieval quality by
  construction.
- The defense targets embedding-near-duplicate content specifically; it is
  not designed to catch the irrelevant (word-shuffled) attack, and its
  effectiveness against contradictory poisoning depends on how close the
  swapped-fact embedding remains to the original.
- Evaluated at a single default poisoning rate and noise level; a full
  robustness curve across rates would need multiple sweep runs (straightforward
  via the config, but not automated into one script).

## Future Work

- Sweep poisoning rate and noise level across a range of values to plot full
  degradation/recovery curves rather than single points
- Compare a cross-encoder re-ranker as a second, stronger defense baseline
- Test on a second dataset (e.g. Natural Questions) to check whether the
  poisoning/defense results generalize beyond SQuAD's structure
- Replace the extractive reader with a generative one and add human or
  LLM-judged answer quality evaluation, in addition to EM/BLEU/ROUGE-L
- Compare defense robustness across multiple embedding models
