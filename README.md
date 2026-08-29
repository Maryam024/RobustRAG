# RobustRAG

An empirical study of corpus poisoning and query embedding noise in retrieval-augmented question answering.

Preprint (Zenodo, DOI): [10.5281/zenodo.22153313](https://zenodo.org/records/22153313)
Author: Maryam Zaheer, Department of Computer Science, UET Lahore

## What this is

This project builds a small, fully reproducible RAG pipeline (BGE encoder + FAISS + an extractive reader) on SQuAD 1.1, and uses it to measure two things side by side: how much a poisoned corpus hurts retrieval and answer quality, and how much noisy query embeddings hurt the same things. It also tests a common lightweight defense (near-duplicate suppression) against the corpus attacks and reports whether it actually helps.

It isn't a new attack or a new defense. It's an evaluation, and the main thing it found is a negative result worth reporting: the defense doesn't recover the loss it's meant to fix, and it's possible to explain exactly why from how the algorithm is built (see Results below).

## Setup

- **Data:** SQuAD 1.1 validation split, 2,067 unique passages after deduplication, 10,570 answerable questions. Validation split specifically, not train, because the reader model (`distilbert-base-cased-distilled-squad`) was fine-tuned on the train split — using it here would let the reader answer from memorized associations instead of the retrieved text.
- **Retriever:** `BAAI/bge-small-en-v1.5` sentence embeddings, FAISS `IndexFlatIP` for exact search. BGE encodes queries with an instruction prefix that passages don't get — easy to miss if you're following a generic embedding tutorial, and it does affect retrieval quality.
- **Poisoning strategies (5% injection rate, 103 documents), all built from real corpus content:**
  - *Near-duplicate* — verbatim copy of a real passage.
  - *Contradictory* — same passage, correct answer swapped for a wrong one pulled from elsewhere in the corpus (falls back to a plain duplicate if the substring swap fails).
  - *Irrelevant* — word-shuffled passage, shares vocabulary but says nothing coherent.
- **Query noise:** Gaussian noise (σ = 0.1) added to query embeddings, then renormalized. Corpus is untouched — this tests a completely separate failure mode.
- **Defense:** near-duplicate suppression. Pulls a candidate pool of 20, keeps candidates in score order as long as they're not too similar (cosine ≥ 0.97) to something already kept, backfills from lower-ranked candidates otherwise.
- **Metrics:** Recall@5, Precision@5, MRR for retrieval; Exact Match, BLEU, ROUGE-L for answer quality (top-1 passage only, extractive reader).

## Results

All numbers are over the full 10,570-question set, seed 42.

| Condition | Recall@5 | Precision@5 | MRR | EM | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| Clean (baseline) | 0.887 | 0.177 | 0.763 | 0.559 | 0.259 | 0.631 |
| Near-duplicate, poisoned | 0.884 | 0.177 | 0.744 | 0.559 | 0.259 | 0.631 |
| Near-duplicate, defended | 0.843 | 0.169 | 0.726 | 0.559 | 0.259 | 0.631 |
| Contradictory, poisoned | 0.883 | 0.177 | 0.750 | 0.558 | 0.258 | 0.629 |
| Contradictory, defended | 0.873 | 0.175 | 0.750 | 0.558 | 0.258 | 0.629 |
| Irrelevant, poisoned | 0.886 | 0.177 | 0.761 | 0.557 | 0.258 | 0.628 |
| Irrelevant, defended | 0.886 | 0.177 | 0.761 | 0.557 | 0.258 | 0.628 |
| Query noise (σ = 0.1) | 0.603 | 0.121 | 0.443 | 0.290 | 0.136 | 0.342 |

**Query noise is the bigger problem.** Recall@5 drops 32% relative to clean, MRR 42%, EM 48%. Everything moves together, which fits noise acting on the query vector directly rather than hitting one specific part of the pipeline.

**Corpus poisoning at 5% is small.** Near-duplicate poisoning leaves EM/BLEU/ROUGE-L unchanged to many decimal places — makes sense, since the poisoned text is an exact copy, so whichever copy wins the ranking slot, the reader gets identical text either way. Contradictory poisoning is the only strategy that changes the actual document content, and it's the only one that moves the answer-quality numbers at all, even if only slightly.

**The defense doesn't work, and it's not a tuning problem.** Suppression drops Recall@5 further in every poisoning condition it touches (near-duplicate: 0.884 → 0.843, a 4.6% relative drop against the poisoned condition; contradictory: 0.883 → 0.873). Meanwhile EM/BLEU/ROUGE-L are identical before and after defense across all conditions. The reason: the algorithm keeps the first (highest-scoring) candidate unconditionally, since there's nothing yet in its kept-list to compare it against. So the top-ranked passage handed to the reader never changes — the defense can only reshuffle ranks 2 through 5, and in these results that reshuffling made retrieval worse, not better. Any defense built the same way (filter a fixed candidate list top-down, never reconsider position 1) will have this same blind spot.

## Running it

```bash
pip install -r requirements.txt
```

Download SQuAD 1.1 and point `config/default.yaml`'s `data.squad_path` at it (a tiny synthetic sample is included under `data/` for sanity-checking the loader without the real dataset).

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

`--skip-answers` runs retrieval only, without loading the reader model.

Config defaults: 2,500-passage subsample (deduplicates to 2,067 unique), top-5 retrieval, 20-candidate defense pool, 5% poisoning rate, σ=0.1 query noise, 0.97 similarity threshold, seed 42.

A minor reproducibility note: repeated runs of the same seeded config showed 4th-decimal-place differences in Recall@5 for the contradictory condition, most likely floating-point summation order in multi-threaded FAISS/PyTorch. Doesn't change any conclusion, just noting it rather than pretending the numbers are perfectly deterministic.

## Limitations

- Everything here is at a single poisoning rate (5%) and a single noise level (σ=0.1). A proper sweep across rates would show whether the defense's negative effect on recall scales with attack strength or is closer to a fixed cost.
- No paired significance testing yet — the poisoning effects especially are small enough that some of the difference could be within noise for a single seeded run. Would need per-query outcome logging (currently only aggregates are kept) to run something like McNemar's test properly.
- The poisoning strategies are heuristic and built from real corpus text, not gradient-optimized adversarial attacks like PoisonedRAG (Zou et al., 2024) or the passage-injection attack in Zhong et al. (2023). Those would likely do more damage, though the defense's failure mode here isn't tied to how the poison was constructed, so I'd expect it to hold either way.
- The reader only sees the top-1 passage and can't hedge or synthesize across passages — a generative reader given multiple passages might behave very differently under the same retrieval failures.
- Recall/Precision/MRR here assume one relevant passage per query, which holds for this SQuAD setup but wouldn't generalize automatically to a corpus with multiple relevant passages per query.

## Future work

- Sweep poisoning rate and noise level instead of reporting one point each.
- Log per-query outcomes and run a proper paired significance test.
- Redesign the defense so it can actually reconsider the top-ranked candidate — e.g. comparing every candidate, including rank 1, against a reference fingerprint of the clean corpus, instead of only comparing candidates against each other.
- Compare against stronger baselines: cross-encoder reranking, semantic clustering, evidence-consistency filtering.
- Try a generative reader instead of the extractive one, and evaluate faithfulness in addition to correctness.
- Repeat on a dataset other than SQuAD and with a different embedding model, to see how much of this generalizes.

## Citation

```bibtex
@misc{zaheer2025robustrag,
  author    = {Zaheer, Maryam},
  title     = {RobustRAG: An Empirical Study of Corpus Poisoning and Query Noise
               in Retrieval-Augmented Question Answering},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22153313},
  url       = {https://zenodo.org/records/22153313}
}
```

## License

MIT — see `LICENSE`.
