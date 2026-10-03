"""Decision logic (plans/03 §5.5), package M1.

`decide` turns a posterior into the `diagnosis` object of 03 §11.1 (without evidence, which
`evidence.py` adds). It gets everything that belongs to other packages as arguments:

    posterior   {class: p} from the Bayes layer (package D1), already built on the masked,
                calibrated model probabilities
    novelty     the object from `Diagnoser.novelty` / `novelty.assess`
    fix_check   callable(top1, top2) -> bool from the fixer (package R1): True when fix(top1)
                alone fails the tests, fix(top2) alone fails, and both together pass
    best_probe  callable(posterior) -> (public probe dict, eig_bits) or None, from the Bayes
                layer's probe selection (03 §6.4)

    if gate != G0                                   -> "gate"
    elif tests all pass:
         top1 == CORRECT or p(top1) < 0.5           -> "correct"
         else                                       -> "correct" with latent = {class, p}
    elif novel                                      -> "novel"
    elif two_bug_check(top1, top2)                  -> "two_bug"
    elif p1 < 0.75 and p1 - p2 < 0.25:
         best probe >= 0.10 bits                    -> "ambiguous", next_probe
         else                                       -> "confident" (band "Possible")
    else                                            -> "confident"

Thresholds come from ml/contracts/params.py.

`top` lists the best class and the runner-up (dropped below 0.02). `twin_set` is set when the
top two belong to one twin set and either they are in the close-call zone (p1 < 0.75 and
p1 - p2 < 0.25) or a probe has already been asked; this matches the fixtures of package W0.
"""
from __future__ import annotations

from ml.contracts.classes import CLASS_INFO, LABELS, MISCONCEPTIONS, band, twin_set_of
from ml.contracts.params import (
    CONFIDENT_MARGIN, CONFIDENT_P, EIG_MIN_BITS, MAX_PROBES, POSTERIOR_STOP, TWO_BUG_P2,
)

LATENT_MIN_P = 0.5              # 03 §5.5: tests passed but the model still names a class at >= 0.5
TOP_MIN_P = 0.02                # a runner-up below this is not listed in `top`


def normalise(posterior):
    """A full {label: p} dict in LABELS order that sums to 1."""
    values = {name: max(float(posterior.get(name, 0.0)), 0.0) for name in LABELS}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("posterior has no probability mass")
    return {name: p / total for name, p in values.items()}


def ranked(posterior):
    return sorted(posterior.items(), key=lambda item: (-item[1], LABELS.index(item[0])))


def top_entry(name, p, force_band=None):
    info = CLASS_INFO[name]
    return {"id": name, "p": round(p, 4), "name": info["name"], "subtitle": info["subtitle"],
            "band": force_band or band(p)}


def decide(posterior, *, gate_code="G0", tests_passed=False, novelty=None, fix_check=None, best_probe=None,
           probes_asked=(), model_version=""):
    probes_asked = list(probes_asked)
    if gate_code != "G0":
        return {"status": "gate", "posterior": {}, "top": [], "twin_set": None, "two_bug": False, "novelty": None,
                "evidence": [], "next_probe": None, "probes_asked": probes_asked, "latent": None,
                "model_version": model_version}

    posterior = normalise(posterior)
    order = ranked(posterior)
    (top1, p1), (top2, p2) = order[0], order[1]
    close_call = p1 < CONFIDENT_P and p1 - p2 < CONFIDENT_MARGIN
    # The twin set is named while the two classes compete, and kept once a probe has settled it;
    # a clear call with a distant runner-up from the same set is not a twin case.
    twin_set = twin_set_of(top1, top2) if (close_call or probes_asked) and p2 >= TOP_MIN_P else None
    result = {
        "status": "confident",
        "posterior": {name: round(p, 4) for name, p in posterior.items()},
        "top": [top_entry(top1, p1)] + ([top_entry(top2, p2)] if p2 >= TOP_MIN_P else []),
        "twin_set": twin_set,
        "two_bug": False, "novelty": novelty, "evidence": [], "next_probe": None,
        "probes_asked": probes_asked, "latent": None, "model_version": model_version,
    }

    if tests_passed:
        result["status"] = "correct"
        if top1 != "CORRECT" and p1 >= LATENT_MIN_P:
            result["latent"] = {"class": top1, "p": round(p1, 4)}
        return result

    if novelty and novelty.get("abstain"):
        result["status"] = "novel"
        return result

    both_fixable = top1 in MISCONCEPTIONS and top2 in MISCONCEPTIONS
    if p2 >= TWO_BUG_P2 and both_fixable and fix_check is not None and fix_check(top1, top2):
        result["status"], result["two_bug"] = "two_bug", True
        return result

    if close_call:
        probe = None
        if best_probe is not None and len(probes_asked) < MAX_PROBES and p1 < POSTERIOR_STOP:
            probe = best_probe(posterior)
        if probe is not None and probe[1] >= EIG_MIN_BITS:
            result["status"] = "ambiguous"
            result["next_probe"] = dict(probe[0], eig_bits=round(float(probe[1]), 4))
        else:
            result["top"][0] = top_entry(top1, p1, force_band="Possible")
    return result
