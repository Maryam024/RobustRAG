import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def _bar_chart(labels: list, values: list, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_retrieval_accuracy(results_by_condition: dict, out_path: str):
    _bar_chart(
        list(results_by_condition.keys()), list(results_by_condition.values()),
        ylabel="Retrieval Accuracy (Recall@k)", title="Retrieval Accuracy by Condition",
        out_path=out_path,
    )


def plot_answer_accuracy(results_by_condition: dict, out_path: str):
    _bar_chart(
        list(results_by_condition.keys()), list(results_by_condition.values()),
        ylabel="Exact Match", title="Answer Accuracy by Condition",
        out_path=out_path,
    )


def plot_poisoning_impact(clean_score: float, scores_by_strategy: dict, out_path: str):
    labels = ["clean"] + list(scores_by_strategy.keys())
    values = [clean_score] + list(scores_by_strategy.values())
    _bar_chart(
        labels, values, ylabel="Retrieval Accuracy (Recall@k)",
        title="Impact of Poisoning Strategy on Retrieval", out_path=out_path,
    )


def plot_defense_recovery(clean_score: float, poisoned_score: float, defended_score: float, out_path: str):
    _bar_chart(
        ["clean", "poisoned", "defended"], [clean_score, poisoned_score, defended_score],
        ylabel="Retrieval Accuracy (Recall@k)", title="Defense Recovery After Near-Duplicate Suppression",
        out_path=out_path,
    )
