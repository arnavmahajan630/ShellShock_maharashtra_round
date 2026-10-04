"""E14: write the exam table for simulated learners (ml_plan/03 §9.3b).

Run: ``.venv\\Scripts\\python -m ml.eval.e14_exam``

Writes ``e14_card.json``, ``e14_table.md`` and ``e14_curve.png`` next to this file.
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
from ml.eval.sim_exam import CAVEAT, POLICIES, SIMULATED_POPULATION, run_exam_sim

HERE = Path(__file__).resolve().parent
CARD_PATH = HERE / "e14_card.json"
TABLE_PATH = HERE / "e14_table.md"
PLOT_PATH = HERE / "e14_curve.png"

# Full stand-in run (seed 1414, 2000 x 20) before the E1 matrix. Simulated learners.
STAND_IN_HEADLINE = {
    "adaptive": 0.481579,
    "fixed": 0.453387,
    "random": 0.417303,
}


def render_table(card):
    lines = [
        "# E14 Exam — simulated learners, not real students",
        "",
        CAVEAT,
        "",
        f"Population: {SIMULATED_POPULATION}.",
        (
            f"n = {card['learners_per_seed']} simulated learners per seed, "
            f"{card['n_seeds']} seeds (seed {card['seed']}). "
            "Cells are the mean and the 95% t interval across seeds."
        ),
        "Items to first true finding: learners with no hidden active class are left out; never found counts as 10.",
        f"Diagnoser noise: {card['diagnoser_noise']['source']}.",
        "",
        "| Policy | Profile-recovery F1 | Brier of P(A) | STABLE classes rechecked | Items to first true finding |",
        "|---|---|---|---|---|",
    ]
    for row in card["rows"]:
        lines.append(
            "| {policy} (simulated) | {f1} | {brier} | {re} | {items} |".format(
                policy=row["policy"],
                f1=fmt_stat(row["profile_f1"]),
                brier=fmt_stat(row["brier"]),
                re=fmt_stat(row["stable_recheck"]),
                items=fmt_stat(row["items_to_first_finding"]),
            )
        )
    acc = card["acceptance"]
    gain = "—" if acc["relative_gain"] is None else f"{acc['relative_gain']:.3f}"
    lines.extend([
        "",
        "Profile-recovery F1 after each item (simulated):",
        "",
        "| Item | adaptive | fixed | random |",
        "|---|---|---|---|",
    ])
    n_items = len(card["curves"]["adaptive"])
    for i in range(n_items):
        cells = []
        for policy in POLICIES:
            cells.append(fmt_stat(card["curves"][policy][i]["profile_f1"]))
        lines.append(f"| {i + 1} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend([
        "",
        (
            f"Final F1, simulated: adaptive {acc['adaptive']:.3f}, fixed {acc['fixed']:.3f}, "
            f"relative gain {gain}. Claim (adaptive > fixed by >= 10% relative) met: {acc['met']}."
        ),
    ])
    comp = card.get("stand_in_comparison")
    if comp:
        moved = ", ".join(
            f"{policy} {comp['profile_f1'][policy]['delta']:+.3f}" for policy in POLICIES
        )
        lines.append(
            "Move versus the stand-in run (simulated, same seed and n): "
            f"{moved}. Relative gain {comp['relative_gain_stand_in']:.3f} -> "
            f"{comp['relative_gain_e1']:.3f} (delta {comp['relative_gain_delta']:+.3f}). "
            f"10% bar on the stand-in run: {comp['ten_percent_bar_stand_in']}; "
            f"on this E1 run: {comp['ten_percent_bar_e1']}."
        )
    lines.append("")
    return "\n".join(lines)


def attach_comparison(card):
    """Record how the full E1 run moved relative to the saved stand-in run."""
    noise = card.get("diagnoser_noise") or {}
    if not str(noise.get("source", "")).startswith("E1"):
        return card
    if card.get("n_seeds") != 20 or card.get("learners_per_seed") != 2000:
        return card
    out = dict(card)
    by_policy = {}
    for policy, old in STAND_IN_HEADLINE.items():
        stat = next(row["profile_f1"] for row in card["rows"] if row["policy"] == policy)
        by_policy[policy] = {
            "stand_in": round(old, 6),
            "e1": round(float(stat["mean"]), 6),
            "delta": round(float(stat["mean"]) - old, 6),
        }
    old_gain = (STAND_IN_HEADLINE["adaptive"] - STAND_IN_HEADLINE["fixed"]) / STAND_IN_HEADLINE["fixed"]
    new_gain = card["acceptance"]["relative_gain"]
    out["stand_in_comparison"] = {
        "population": SIMULATED_POPULATION,
        "note": (
            "Previous full E14 run used the stand-in matrix. Same seed 1414, "
            "2000 simulated learners per seed, 20 seeds."
        ),
        "profile_f1": by_policy,
        "relative_gain_stand_in": round(old_gain, 6),
        "relative_gain_e1": None if new_gain is None else round(float(new_gain), 6),
        "relative_gain_delta": None if new_gain is None else round(float(new_gain) - old_gain, 6),
        "ten_percent_bar_stand_in": bool(old_gain >= 0.10),
        "ten_percent_bar_e1": bool(card["acceptance"]["met"]),
    }
    return out


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
        for key in ("profile_f1", "brier", "stable_recheck", "items_to_first_finding", "mean_items"):
            copied[key] = _round_stat(row[key])
        out["rows"].append(copied)
    curves = {}
    for policy, series in card["curves"].items():
        curves[policy] = []
        for point in series:
            curves[policy].append({
                "item": point["item"],
                "profile_f1": _round_stat(point["profile_f1"]),
                "simulated": True,
            })
    out["curves"] = curves
    acc = dict(card["acceptance"])
    for key in ("adaptive", "fixed", "relative_gain"):
        if acc[key] is not None:
            acc[key] = round(acc[key], 6)
    out["acceptance"] = acc
    out["plots"] = [PLOT_PATH.name]
    return out


def write_plot(card, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"adaptive": "#54A24B", "fixed": "#4C78A8", "random": "#F58518"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for policy in POLICIES:
        xs = []
        ys = []
        lo = []
        hi = []
        for point in card["curves"][policy]:
            stat = point["profile_f1"]
            xs.append(point["item"])
            ys.append(stat["mean"])
            lo.append(stat["lo"])
            hi.append(stat["hi"])
        ax.plot(xs, ys, label=policy, color=colors[policy], marker="o")
        ax.fill_between(xs, lo, hi, color=colors[policy], alpha=0.15)
    ax.set_xlabel("items asked")
    ax.set_ylabel("profile-recovery F1")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("E14 on simulated learners (not real students)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_outputs(card, directory=None):
    root = HERE if directory is None else Path(directory)
    card = attach_comparison(card)
    payload = card_for_disk(card)
    (root / CARD_PATH.name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table = render_table(card)
    (root / TABLE_PATH.name).write_text(table, encoding="utf-8")
    write_plot(card, root / PLOT_PATH.name)
    return table


def main():
    card = run_exam_sim()
    table = write_outputs(card)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print(table, flush=True)


if __name__ == "__main__":
    main()
