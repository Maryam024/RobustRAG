import argparse
import json
from pathlib import Path

from src.utils.config import load_config, resolve_path
from src.utils.logging import get_logger
from src.data.loader import load_squad_corpus
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.attacks.poisoning import PoisoningEngine
from src.attacks.noise import add_query_noise
from src.defense.filter import suppress_near_duplicates
from src.evaluation.evaluator import evaluate_retrieval, evaluate_answers
from src.evaluation.reader import ExtractiveReader
from src.evaluation import visualize as viz

logger = get_logger("run")


def _load_and_index(config: dict):
    # shared setup for every experiment
    data_cfg, emb_cfg = config["data"], config["embedding"]

    corpus, queries = load_squad_corpus(
        path=resolve_path(config, data_cfg["squad_path"]),
        num_passages=data_cfg["num_passages"],
        seed=data_cfg["seed"],
    )
    logger.info(f"Corpus: {len(corpus)} passages, {len(queries)} evaluable queries")

    embedding_model = EmbeddingModel(
        model_name=emb_cfg["model_name"],
        batch_size=emb_cfg["batch_size"],
        device=emb_cfg["device"],
    )
    retriever = Retriever(embedding_model)
    retriever.build_index(corpus)
    return retriever, corpus, queries


def _get_reader(config: dict, args):
    if args.skip_answers:
        return None
    return ExtractiveReader(config["evaluation"]["reader_model"])


def _score(queries, results, k, reader):
    scores = evaluate_retrieval(queries, results, k)
    if reader is not None:
        scores.update(evaluate_answers(queries, results, reader))
    return scores


def _save_log(config: dict, name: str, payload: dict):
    log_dir = Path(resolve_path(config, config["output"]["results_dir"])) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved {out_path}")


def _log_scores(label: str, scores: dict):
    parts = ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
    logger.info(f"[{label}] {parts}")


def run_baseline(config: dict, args):
    retriever, _, queries = _load_and_index(config)
    k = config["retrieval"]["top_k"]
    reader = _get_reader(config, args)

    results = retriever.retrieve([q.question for q in queries], k=k)
    scores = _score(queries, results, k, reader)
    _log_scores("baseline", scores)
    _save_log(config, "baseline", scores)


def run_poisoning(config: dict, args, reader=None):
    # reused by run_defense to avoid loading the reader twice
    strategy = args.strategy or config["attacks"]["poisoning"]["strategy"]
    retriever, corpus, queries = _load_and_index(config)
    k = config["retrieval"]["top_k"]
    if reader is None:
        reader = _get_reader(config, args)

    clean_results = retriever.retrieve([q.question for q in queries], k=k)
    clean_scores = _score(queries, clean_results, k, reader)
    _log_scores("clean", clean_scores)

    rate = config["attacks"]["poisoning"]["rate"]
    engine = PoisoningEngine(strategy=strategy, rate=rate, seed=config["data"]["seed"])
    poisoned_corpus, poisoned_ids = engine.apply(corpus, queries)
    logger.info(f"Injected {len(poisoned_ids)} poisoned passages ({strategy})")

    retriever.build_index(poisoned_corpus)
    poisoned_results = retriever.retrieve([q.question for q in queries], k=k)
    poisoned_scores = _score(queries, poisoned_results, k, reader)
    _log_scores("poisoned", poisoned_scores)

    payload = {"strategy": strategy, "clean": clean_scores, "poisoned": poisoned_scores}
    _save_log(config, f"poisoning_{strategy}", payload)
    return retriever, queries, payload


def run_noise(config: dict, args):
    retriever, _, queries = _load_and_index(config)
    k = config["retrieval"]["top_k"]
    reader = _get_reader(config, args)
    questions = [q.question for q in queries]

    clean_embeddings = retriever.embedding_model.encode_queries(questions)
    clean_results = retriever.retrieve_from_embeddings(clean_embeddings, k=k)
    clean_scores = _score(queries, clean_results, k, reader)
    _log_scores("clean", clean_scores)

    level = config["attacks"]["noise"]["level"]
    noisy_embeddings = add_query_noise(clean_embeddings, level=level, seed=config["data"]["seed"])
    noisy_results = retriever.retrieve_from_embeddings(noisy_embeddings, k=k)
    noisy_scores = _score(queries, noisy_results, k, reader)
    _log_scores("noisy", noisy_scores)

    payload = {"level": level, "clean": clean_scores, "noisy": noisy_scores}
    _save_log(config, "noise", payload)


def run_defense(config: dict, args):
    reader = _get_reader(config, args)
    retriever, queries, poisoning_payload = run_poisoning(config, args, reader=reader)
    strategy = poisoning_payload["strategy"]
    k = config["retrieval"]["top_k"]
    defense_cfg = config["defense"]

    questions = [q.question for q in queries]
    candidate_results = retriever.retrieve(questions, k=defense_cfg["candidate_pool_size"])
    defended_results = [
        suppress_near_duplicates(
            candidates, retriever.doc_embeddings, k=k,
            similarity_threshold=defense_cfg["similarity_threshold"],
        )
        for candidates in candidate_results
    ]
    defended_scores = _score(queries, defended_results, k, reader)
    _log_scores("defended", defended_scores)

    payload = dict(poisoning_payload, defended=defended_scores)
    _save_log(config, f"defense_{strategy}", payload)


def run_report(config: dict, args):
    import pandas as pd

    log_dir = Path(resolve_path(config, config["output"]["results_dir"])) / "logs"
    fig_dir = Path(resolve_path(config, config["output"]["results_dir"])) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def load(name):
        path = log_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    baseline = load("baseline")
    strategies = ["near_duplicate", "contradictory", "irrelevant"]
    poisoning_logs = {s: load(f"poisoning_{s}") for s in strategies}
    defense_logs = {s: load(f"defense_{s}") for s in strategies}

    if baseline:
        viz.plot_retrieval_accuracy(
            {"clean": baseline["retrieval_accuracy"]}, str(fig_dir / "retrieval_accuracy.png"))
        if "exact_match" in baseline:
            viz.plot_answer_accuracy(
                {"clean": baseline["exact_match"]}, str(fig_dir / "answer_accuracy.png"))

    available_poisoning = {s: p for s, p in poisoning_logs.items() if p}
    if available_poisoning:
        clean_ref = next(iter(available_poisoning.values()))["clean"]["retrieval_accuracy"]
        by_strategy = {s: p["poisoned"]["retrieval_accuracy"] for s, p in available_poisoning.items()}
        viz.plot_poisoning_impact(clean_ref, by_strategy, str(fig_dir / "poisoning_impact.png"))

    available_defense = {s: p for s, p in defense_logs.items() if p}
    for strategy, payload in available_defense.items():
        viz.plot_defense_recovery(
            payload["clean"]["retrieval_accuracy"],
            payload["poisoned"]["retrieval_accuracy"],
            payload["defended"]["retrieval_accuracy"],
            str(fig_dir / f"defense_recovery_{strategy}.png"),
        )

    logger.info(f"Figures written to {fig_dir}")

    rows = []
    if baseline:
        rows.append({"condition": "baseline", **baseline})
    for strategy, payload in available_poisoning.items():
        rows.append({"condition": f"poisoning_{strategy}_clean", **payload["clean"]})
        rows.append({"condition": f"poisoning_{strategy}_poisoned", **payload["poisoned"]})
    for strategy, payload in available_defense.items():
        rows.append({"condition": f"defense_{strategy}_defended", **payload["defended"]})

    if rows:
        results_dir = Path(resolve_path(config, config["output"]["results_dir"]))
        summary_path = results_dir / "results_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        logger.info(f"Summary table written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="RobustRAG experiment runner")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--experiment", default="baseline",
        choices=["baseline", "poisoning", "noise", "defense", "report"],
    )
    parser.add_argument(
        "--strategy", default=None, choices=["near_duplicate", "contradictory", "irrelevant"],
        help="Overrides config's attacks.poisoning.strategy for poisoning/defense experiments.",
    )
    parser.add_argument(
        "--skip-answers", action="store_true",
        help="Skip the extractive-reader answer metrics (EM/BLEU/ROUGE-L) and only report retrieval metrics.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    experiments = {
        "baseline": run_baseline,
        "poisoning": run_poisoning,
        "noise": run_noise,
        "defense": run_defense,
        "report": run_report,
    }
    experiments[args.experiment](config, args)


if __name__ == "__main__":
    main()
