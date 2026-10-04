"""E10: write the resolution table for simulated learners (ml_plan/03 §9.3).

Run: ``.venv\\Scripts\\python -m ml.eval.e10_resolution``

Writes ``e10_card.json``, ``e10_table.md`` and ``e10_bars.png`` next to this file.
Every number is from simulated learners, not real students.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

from ml.eval._v1_stats import fmt_stat
from ml.eval.sim_learners import CAVEAT, LEARNER_TYPES, POLICIES, SIMULATED_POPULATION, run_resolution

HERE = Path(__file__).resolve().parent
CARD_PATH = HERE / "e10_card.json"
TABLE_PATH = HERE / "e10_table.md"
PLOT_PATH = HERE / "e10_bars.png"


def render_table(card):
    lines = [
        "# E10 Resolution — simulated learners, not real students",
        "",
        CAVEAT,
        "",
        f"Population: {SIMULATED_POPULATION}.",
        (
            f"n = {card['learners_per_type_per_seed']} simulated learners per type per seed, "
            f"{card['n_seeds']} seeds (seed {card['seed']}). "
            "Cells are the mean and the 95% t interval across seeds."
        ),
        "A dash means the conditioning set is empty for that type (nobody is active, or nobody is inactive).",
        "",
        "| Policy | Simulated learner type | False-resolve | False-not-yet | Items to decision |",
        "|---|---|---|---|---|",
    ]
    for row in card["rows"]:
        lines.append(
            "| {policy} | {type} (simulated) | {fr} | {fn} | {items} |".format(
                policy=row["policy"],
                type=row["type"],
                fr=fmt_stat(row["false_resolve"]),
                fn=fmt_stat(row["false_not_yet"]),
                items=fmt_stat(row["items_to_decision"]),
            )
        )
    acc = card["acceptance"]
    ratio = "—" if acc["ratio"] is None else f"{acc['ratio']:.3f}"
    lines.extend([
        "",
        (
            f"Pattern-copier false-resolve, simulated: ours {acc['ours']:.3f}, "
            f"naive {acc['naive']:.3f}, ratio {ratio}. "
            f"Claim (ours ≤ ½ × naive) met: {acc['met']}."
        ),
        "",
    ])
    return "\n".join(lines)


def _round_stat(stat):
    if stat is None:
        return None

    def clean(value):
        value = round(float(value), 6)
        return 0.0 if value == 0 else value

    mean = clean(stat["mean"])
    lo = clean(stat["lo"])
    hi = clean(stat["hi"])
    return {"mean": mean, "lo": lo, "hi": hi, "n_seeds": stat["n_seeds"]}


def card_for_disk(card):
    out = dict(card)
    out["rows"] = []
    for row in card["rows"]:
        copied = dict(row)
        for key in ("false_resolve", "false_not_yet", "items_to_decision"):
            copied[key] = _round_stat(row[key])
        out["rows"].append(copied)
    acc = dict(card["acceptance"])
    for key in ("ours", "naive", "ratio"):
        if acc[key] is not None:
            acc[key] = round(acc[key], 6)
    out["acceptance"] = acc
    out["plots"] = [PLOT_PATH.name]
    return out


def write_plot(card, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = (
        (axes[0], "false_resolve", "False-resolve"),
        (axes[1], "false_not_yet", "False-not-yet"),
    )
    colors = {"naive": "#4C78A8", "4-check": "#F58518", "ours": "#54A24B"}
    width = 0.25
    for ax, field, title in panels:
        def cell(learner_type, policy):
            return next(
                row[field] for row in card["rows"]
                if row["type"] == learner_type and row["policy"] == policy
            )

        present = [
            learner_type for learner_type in LEARNER_TYPES
            if all(cell(learner_type, policy) is not None for policy in POLICIES)
        ]
        x = list(range(len(present)))
        for p_i, policy in enumerate(POLICIES):
            means = []
            yerr_lo = []
            yerr_hi = []
            for learner_type in present:
                stat = cell(learner_type, policy)
                means.append(stat["mean"])
                yerr_lo.append(max(0.0, stat["mean"] - stat["lo"]))
                yerr_hi.append(max(0.0, stat["hi"] - stat["mean"]))
            positions = [i + (p_i - 1) * width for i in x]
            ax.bar(
                positions, means, width=width, label=policy, color=colors[policy],
                yerr=[yerr_lo, yerr_hi], capsize=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("rate")
    axes[0].legend(frameon=False)
    fig.suptitle("E10 on simulated learners (not real students)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_outputs(card, directory=None):
    root = HERE if directory is None else Path(directory)
    card_path = root / CARD_PATH.name
    table_path = root / TABLE_PATH.name
    plot_path = root / PLOT_PATH.name
    payload = card_for_disk(card)
    card_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table = render_table(card)
    table_path.write_text(table, encoding="utf-8")
    write_plot(card, plot_path)
    return table


def main():
    card = run_resolution()
    table = write_outputs(card)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print(table, flush=True)


if __name__ == "__main__":
    main()
