# RobustRAG: Evaluating Retrieval Robustness Against Corpus Poisoning

A research-oriented evaluation framework for studying the robustness of Retrieval-Augmented Generation (RAG) pipelines under **corpus poisoning** and **query-side retrieval noise**.

RobustRAG evaluates how different perturbations affect both **retrieval quality** and **downstream answer quality**, and investigates whether a lightweight embedding-based defense can mitigate degradation caused by poisoning.

> **Research scope:** RobustRAG is an evaluation study rather than a novel retrieval or defense algorithm. The goal is to provide a controlled, reproducible framework for measuring RAG robustness and identifying failure modes under different perturbation conditions.

---

## Research Questions

This project investigates three main questions:

1. **How sensitive is RAG retrieval to different forms of corpus poisoning?**
2. **How severely does query-side embedding noise affect retrieval and answer quality?**
3. **Can a lightweight near-duplicate suppression strategy recover performance after corpus poisoning?**

The experiments isolate corpus-level and query-level perturbations so that their effects can be analyzed independently.

---

## Motivation

RAG systems depend heavily on the quality of the retrieval corpus. If an attacker can inject additional documents into the corpus, retrieved results may become less reliable even when the original evidence remains available.

This study evaluates three corpus-poisoning scenarios:

* **Near-duplicate content** that competes with genuine passages
* **Contradictory content** that preserves the original topic while introducing an incorrect answer
* **Irrelevant but vocabulary-overlapping content** that introduces noisy retrieval candidates

The project also studies a separate failure mode: **query-side embedding noise**. In this setting, the corpus remains unchanged while Gaussian perturbations are applied directly to query embeddings.

This separation allows the experiments to distinguish between robustness to **corpus contamination** and robustness to **query representation degradation**.

---

## Project Overview

The evaluation pipeline consists of:

1. Load a SQuAD-derived retrieval corpus
2. Encode passages using a sentence-embedding model
3. Build a FAISS retrieval index
4. Establish clean-corpus baseline performance
5. Inject corpus-poisoning documents using three attack strategies
6. Measure retrieval and downstream answer-quality changes
7. Independently perturb query embeddings with Gaussian noise
8. Apply a lightweight near-duplicate suppression defense to poisoned retrieval results
9. Compare clean, attacked, and defended conditions
10. Generate summary metrics and research figures

All experiments are controlled through `run.py` and a YAML configuration file.

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

### Main Components

| Component     | Purpose                                            |
| ------------- | -------------------------------------------------- |
| `data/`       | Dataset loading and corpus construction            |
| `retrieval/`  | Sentence embeddings, FAISS indexing, and retrieval |
| `attacks/`    | Corpus poisoning and query-noise generation        |
| `defense/`    | Near-duplicate suppression                         |
| `evaluation/` | Retrieval metrics, extractive QA, and reporting    |
| `run.py`      | Unified experiment entry point                     |

---

## Methodology

### Embedding Model

Retrieval uses:

`BAAI/bge-small-en-v1.5`

through `sentence-transformers`.

The model was selected as a lightweight sentence-embedding model suitable for semantic text retrieval and CPU-based experimentation.

### Corpus

The retrieval corpus is constructed from SQuAD paragraphs.

Each answerable question provides:

* a query
* a gold passage
* an expected answer

This enables direct evaluation of retrieval using Recall@K, Precision@K, and MRR, while the downstream extractive reader provides answer-quality metrics.

### Corpus Poisoning

Three poisoning strategies are evaluated.

#### 1. Near-Duplicate

Real corpus passages are duplicated and injected into the corpus.

The original passage remains available, allowing the experiment to measure the effect of additional competing documents.

#### 2. Contradictory

A real passage is modified so that its original answer is replaced with an incorrect answer while retaining the surrounding topical context.

This creates a semantically similar passage containing conflicting information.

#### 3. Irrelevant

A real passage is transformed through word shuffling to preserve some vocabulary overlap while substantially reducing its coherent semantic content.

---

## Query-Side Noise

Query noise is evaluated independently from corpus poisoning.

Gaussian perturbations are applied directly to the query embeddings and the resulting vectors are renormalized before retrieval.

This experiment asks a separate robustness question:

> How much does RAG performance deteriorate when the query representation itself becomes noisy?

---

## Defense

The evaluated defense is a lightweight **near-duplicate suppression mechanism**.

It retrieves an expanded candidate pool and greedily removes candidates whose embeddings are sufficiently similar to candidates already selected.

The default cosine-similarity threshold is:

```text
0.97
```

The defense is intentionally simple and interpretable. It is evaluated as a baseline mitigation strategy rather than presented as a novel defense algorithm.

---

# Experimental Results

The current experiments reveal substantially different robustness behavior across perturbation types.

### Clean Baseline

The clean retrieval condition achieves:

| Metric             |      Clean |
| ------------------ | ---------: |
| Recall@K           | **0.8871** |
| Precision@K        | **0.1774** |
| MRR                | **0.7634** |
| Retrieval Accuracy | **0.8871** |
| Exact Match        | **0.5592** |
| BLEU               | **0.2588** |
| ROUGE-L            | **0.6305** |

---

## Corpus Poisoning

The current poisoning experiments produce relatively small changes compared with the query-noise experiment.

| Condition      | Recall@K |    MRR | Exact Match |
| -------------- | -------: | -----: | ----------: |
| Clean          |   0.8871 | 0.7634 |      0.5592 |
| Near-Duplicate |   0.8836 | 0.7440 |      0.5592 |
| Contradictory  |   0.8830 | 0.7546 |      0.5579 |
| Irrelevant     |   0.8862 | 0.7614 |      0.5570 |

The current results indicate that the evaluated poisoning rate causes **modest aggregate degradation** in this experimental configuration.

Near-duplicate poisoning produces the largest MRR decrease among the three evaluated poisoning strategies, while irrelevant poisoning has the smallest effect on retrieval metrics.

These results should be interpreted as preliminary rather than as evidence that the evaluated poisoning strategies are universally ineffective. Statistical testing and experiments across multiple poisoning rates are needed to determine the robustness and significance of these effects.

---

## Query Noise

The query-noise experiment produces a substantially larger degradation.

| Metric      |  Clean |  Noisy | Relative Change |
| ----------- | -----: | -----: | --------------: |
| Recall@K    | 0.8871 | 0.6030 |      **−32.0%** |
| MRR         | 0.7634 | 0.4432 |      **−41.9%** |
| Exact Match | 0.5592 | 0.2898 |      **−48.2%** |
| BLEU        | 0.2588 | 0.1358 |      **−47.5%** |
| ROUGE-L     | 0.6305 | 0.3423 |      **−45.7%** |

The degradation is consistent across retrieval and answer-quality metrics.

This suggests that, under the current experimental configuration, **query representation noise is substantially more damaging than the evaluated corpus-poisoning conditions**.

---

## Defense Evaluation

The current defense experiment does **not** demonstrate recovery of downstream answer quality.

For the contradictory poisoning condition:

| Metric      |  Clean | Poisoned | Defended |
| ----------- | -----: | -------: | -------: |
| Recall@K    | 0.8871 |   0.8831 |   0.8728 |
| MRR         | 0.7634 |   0.7549 |   0.7504 |
| Exact Match | 0.5592 |   0.5582 |   0.5582 |
| BLEU        | 0.2588 |   0.2581 |   0.2581 |
| ROUGE-L     | 0.6305 |   0.6289 |   0.6289 |

In this experiment, the defense does not improve Exact Match, BLEU, or ROUGE-L relative to the poisoned condition.

Recall@K and MRR also decrease after defense.

Therefore, the current implementation should be interpreted as a **negative defense result / limitation**, rather than as evidence of successful mitigation.

This result motivates further investigation into the interaction between candidate suppression, ranking, and downstream top-1 answer extraction.

---

# Research Findings

The current experiments support three preliminary observations:

### 1. Query noise causes substantial degradation

Gaussian perturbation of query embeddings produces large and consistent drops in both retrieval and answer-quality metrics.

The largest observed effects occur in Exact Match, BLEU, and ROUGE-L, all of which decrease by approximately 45–48%.

### 2. The evaluated poisoning attacks have smaller aggregate effects

At the current poisoning rate, near-duplicate, contradictory, and irrelevant attacks produce relatively modest changes in aggregate retrieval metrics.

This does not establish that corpus poisoning is harmless; rather, it indicates that **attack intensity and attack construction require further systematic evaluation**.

### 3. The lightweight defense does not currently recover performance

The evaluated defense fails to improve downstream answer metrics in the current contradictory-poisoning experiment and slightly reduces retrieval metrics.

This provides a concrete failure case for the defense and motivates investigation into stronger mitigation strategies.

---

# Experimental Reproducibility

All experiment parameters are defined in:

```text
config/default.yaml
```

Default settings include:

* Corpus size: 2,500 passages
* Retrieval depth: top-5
* Poisoning rate: 5%
* Query noise level: 0.1
* Defense similarity threshold: 0.97
* Random seed: 42

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the SQuAD dataset and configure its path in:

```text
config/default.yaml
```

Run the experiments:

```bash
python run.py --experiment baseline

python run.py --experiment poisoning --strategy near_duplicate

python run.py --experiment poisoning --strategy contradictory

python run.py --experiment poisoning --strategy irrelevant

python run.py --experiment noise

python run.py --experiment defense --strategy contradictory

python run.py --experiment report
```

Use `--skip-answers` to evaluate retrieval without running the extractive reader.

The random seed controls the corpus subsampling, poisoning generation, and noise generation. Exact reproducibility may additionally depend on underlying numerical-library and hardware behavior.

---

# Evaluation Metrics

### Retrieval

**Recall@K**
Measures whether the gold passage appears within the top-K retrieved results.

**Precision@K**
Measures the proportion of retrieved passages that are relevant.

**MRR**
Measures how highly the first relevant passage is ranked.

### Answer Quality

The current pipeline uses:

* Exact Match
* BLEU
* ROUGE-L

A lightweight extractive QA reader,

```text
distilbert-base-cased-distilled-squad
```

is applied to the top-ranked retrieved passage.

Using an extractive reader keeps the current evaluation focused on the relationship between retrieval quality and answer extraction without introducing a separate generative LLM.

---

# Limitations

* Poisoning strategies are heuristic constructions rather than optimized adversarial attacks.
* The current poisoning experiments use a single default poisoning rate.
* The current noise experiment uses a single noise level.
* Statistical significance testing has not yet been performed for the aggregate poisoning results.
* Recall@K, Precision@K, and MRR assume one relevant passage per query in this SQuAD-derived setup.
* The reader receives only the top-1 retrieved passage rather than a multi-passage context.
* The defense specifically targets embedding-level similarity and is not designed as a general-purpose poisoning defense.
* Contradictory poisoning depends on successfully modifying the answer-bearing content; fallback cases should be monitored when reproducing the experiment.
* Results are currently based on a SQuAD-derived corpus and require evaluation on additional datasets to assess generalization.
* Reproducibility can be affected by numerical-library or hardware-level nondeterminism despite fixed random seeds.

---

# Future Work

The next experimental steps are:

### 1. Poisoning-rate sweep

Evaluate:

```text
0% → 5% → 10% → 20% → 30%
```

to determine whether degradation follows a consistent dose-response relationship.

### 2. Noise-level sweep

Evaluate multiple query-noise levels and construct robustness curves rather than reporting a single perturbation strength.

### 3. Statistical significance testing

Use per-query predictions to perform paired statistical tests and determine whether observed poisoning effects are statistically significant.

### 4. Defense analysis

Investigate why the current suppression mechanism does not recover downstream answer quality and compare it with stronger reranking or filtering baselines.

### 5. Additional datasets

Evaluate the framework on datasets beyond SQuAD to determine whether the observed robustness patterns generalize.

### 6. Stronger defense baselines

Compare lightweight suppression against:

* Cross-encoder reranking
* Semantic clustering
* Evidence consistency filtering
* Other retrieval-time filtering strategies

### 7. Generative RAG evaluation

Extend the current extractive setup to a generative reader and evaluate answer faithfulness and correctness in addition to retrieval metrics.

### 8. Multiple embedding models

Repeat the experiments using different embedding models to determine whether robustness is dependent on the representation model.

---

# Research Status

RobustRAG is currently an **experimental evaluation study**.

The current results demonstrate a substantial sensitivity to query-side embedding noise, while the evaluated corpus-poisoning attacks produce comparatively modest degradation at the tested poisoning rate. The lightweight defense does not currently demonstrate performance recovery.

Further experiments involving **attack-rate sweeps, statistical significance testing, stronger defense baselines, and cross-dataset evaluation** are required before drawing broader conclusions about RAG robustness.

---

## License

See `LICENSE` for details.
