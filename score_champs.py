"""CHAMPS KNOW adherence scorer — the instrument for honing the discipline.

Reads saved run JSON(s) and produces a per-principle scorecard, so a prompt/schema
change can be judged by whether the agent actually *exhibited* each principle —
fast, cheap signal that doesn't wait for markets to resolve. Most principles are
scored mechanically from the saved telemetry; the soft ones (Pre-mortem,
Synthesize, No-sacred-cows) use a cheap DeepSeek-flash judge.

  uv run score_champs.py runs/<file>.json            # one run
  uv run score_champs.py eval_runs/baseline          # a whole batch (dir)
  uv run score_champs.py eval_runs/baseline --no-judge  # mechanical only (free)

Scores are 0..1 per principle. K (Keep score) and W (Wisdom of crowds) are
system-level (calibration + multi-trial aggregation), not per-run, so they're not
scored here.
"""

import argparse
import glob
import json
import math
import os
import statistics

import forecaster

JUDGE_MODEL = "deepseek-v4-flash"  # cheap tier for the soft-principle judge


def _logit(p: float) -> float:
    p = min(0.95, max(0.05, float(p)))
    return math.log(p / (1 - p))


def _grounding(trial: dict) -> tuple[int, int, int, int]:
    """(total, grounded, prior, fabricated) reference cases — recomputed from raw
    data (source_id vs the run's real search registry) so it works on old runs."""
    valid = set((trial.get("sources") or {}).keys())
    total = grounded = prior = fab = 0
    for c in trial.get("reference_cases") or []:
        if not isinstance(c, dict):
            continue
        total += 1
        sid = str(c.get("source_id", "")).strip()
        if sid in valid:
            grounded += 1
        elif sid in ("prior", "unverified"):
            prior += 1 if sid == "prior" else 0
            fab += 1 if sid == "unverified" else 0
        else:
            fab += 1
    return total, grounded, prior, fab


def score_trial_mechanical(trial: dict) -> dict:
    """Per-principle mechanical sub-scores (0..1) + the raw basis, for one trial."""
    steps = trial.get("steps") or []
    probs = [s.get("probability") for s in steps if s.get("probability") is not None]
    st = trial.get("stats") or {}

    # C — a non-trivial comparison class is named.
    cc = (trial.get("comparison_class") or "").strip()
    c = 1.0 if len(cc) >= 15 else (0.5 if cc else 0.0)

    # O — outside view: base rate present AND cases researched (grounded), not assumed.
    total, grounded, prior, fab = _grounding(trial)
    grounded_ratio = grounded / total if total else 0.0
    anchor = 1.0 if trial.get("base_rate") is not None else 0.0
    o = 0.5 * anchor + 0.5 * grounded_ratio

    # H — hunt: depth of search/read activity.
    n_searches = st.get("n_searches", 0) or 0
    n_reads = st.get("n_reads", 0) or 0
    domains = len({(v.get("url") or "").split("/")[2] for v in (trial.get("sources") or {}).values() if v.get("url")})
    h = min(1.0, (n_searches + n_reads) / 6.0)

    # A — adjust often: several incremental updates, no single wild lurch.
    n_updates = sum(1 for a, b in zip(probs, probs[1:]) if abs(a - b) > 1e-9)
    max_jump = max((abs(_logit(b) - _logit(a)) for a, b in zip(probs, probs[1:])), default=0.0)
    if n_updates >= 2 and max_jump <= 2.0:
        a = 1.0
    elif n_updates >= 1 and max_jump <= 2.5:
        a = 0.6
    else:
        a = 0.3
    if max_jump > 3.0:           # a near-extreme single lurch
        a = min(a, 0.3)

    # M — make precise: penalize coarse round numbers.
    p = trial.get("probability")
    m = 1.0 if (p is not None and abs(round(float(p), 2) - round(float(p), 1)) > 1e-9) else 0.5

    return {
        "C": round(c, 2), "O": round(o, 2), "H": round(h, 2), "A": round(a, 2), "M": round(m, 2),
        "_raw": {
            "comparison_class": cc[:80],
            "ref_cases": f"{grounded} grounded / {prior} prior / {fab} fabricated of {total}",
            "searches": n_searches, "reads": n_reads, "domains": domains,
            "n_updates": n_updates, "max_logit_jump": round(max_jump, 2),
            "probability": p, "fabricated": fab,
        },
    }


JUDGE_PROMPT = (
    "You are auditing a superforecaster's run for adherence to three CHAMPS KNOW "
    "principles. Score each 0-2 (0=absent, 1=partial, 2=strong) and give a one-line "
    "reason. Return ONLY a JSON object: "
    '{"P":{"score":int,"why":str},"S":{"score":int,"why":str},"N":{"score":int,"why":str}}\n'
    "- P (Pre-mortem): does the final justification state the strongest case the "
    "forecast is WRONG and price it in (not a token sentence)?\n"
    "- S (Synthesize): does it integrate multiple independent sources/perspectives "
    "into one view, rather than leaning on a single narrative?\n"
    "- N (No sacred cows): is it driven by evidence rather than ideology / wishful "
    "thinking / a sacred assumption?\n\n"
)


def score_trial_judge(question: str, trial: dict) -> dict:
    """Soft principles via a cheap flash judge. Returns {P,S,N} as 0..1 (or zeros on failure)."""
    payload = {
        "question": question,
        "final_probability": trial.get("probability"),
        "submit_justification_and_premortem": trial.get("action_input"),
        "evidence_for": (trial.get("evidence_for") or [])[:10],
        "evidence_against": (trial.get("evidence_against") or [])[:10],
        "update_reasoning": trial.get("update_reasoning"),
    }
    prov = forecaster.get_provider()
    try:
        text, _ = prov.complete(
            model=JUDGE_MODEL, system=None,
            prompt=JUDGE_PROMPT + "RUN:\n" + json.dumps(payload, ensure_ascii=False)[:6000],
            temperature=0.0, max_output_tokens=600,
        )
        obj = forecaster._parse_json_object(text) or {}
        return {k: (obj.get(k, {}).get("score", 0) or 0) / 2.0 for k in ("P", "S", "N")}, \
               {k: obj.get(k, {}).get("why", "") for k in ("P", "S", "N")}
    except Exception as exc:
        return {"P": 0.0, "S": 0.0, "N": 0.0}, {"_error": str(exc)}


def score_run(path: str, judge: bool) -> dict:
    d = json.load(open(path))
    trials = d.get("trials") or []
    if not trials:
        return {"file": os.path.basename(path), "error": "no trials"}
    mech = [score_trial_mechanical(t) for t in trials]
    avg = {k: round(statistics.mean(m[k] for m in mech), 2) for k in ("C", "O", "H", "A", "M")}
    why = {}
    if judge:
        forecaster.set_provider("deepseek")
        js, jw = score_trial_judge(d.get("question", ""), trials[0])  # judge representative trial
        avg.update({k: round(v, 2) for k, v in js.items()})
        why = jw
    return {
        "file": os.path.basename(path),
        "question": (d.get("question") or "")[:70],
        "provider": d.get("provider"),
        "scores": avg,
        "raw": mech[0]["_raw"],
        "judge_why": why,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score CHAMPS KNOW adherence of saved runs")
    ap.add_argument("target", help="a run .json file, or a directory of them")
    ap.add_argument("--no-judge", action="store_true", help="mechanical only (skip the flash judge for P/S/N)")
    ap.add_argument("--out", default=None, help="write the full scorecard JSON here")
    args = ap.parse_args()

    files = (sorted(glob.glob(os.path.join(args.target, "*.json")))
             if os.path.isdir(args.target) else [args.target])
    judge = not args.no_judge
    cards = [score_run(f, judge) for f in files]
    cards = [c for c in cards if "scores" in c]

    principles = ["C", "O", "H", "A", "M"] + (["P", "S", "N"] if judge else [])
    print(f"\n{'question':<52} " + " ".join(f"{p:>4}" for p in principles))
    print("-" * (52 + 5 * len(principles)))
    for c in cards:
        row = " ".join(f"{c['scores'].get(p, 0):>4.2f}" for p in principles)
        print(f"{c['question']:<52} {row}")
        print(f"    └ {c['raw']['ref_cases']} | {c['raw']['searches']}s/{c['raw']['reads']}r/"
              f"{c['raw']['domains']}dom | {c['raw']['n_updates']} updates, max jump "
              f"{c['raw']['max_logit_jump']} | p={c['raw']['probability']}"
              + (f" | ⚠ {c['raw']['fabricated']} fabricated" if c['raw']['fabricated'] else ""))

    if cards:
        batch = {p: round(statistics.mean(c["scores"].get(p, 0) for c in cards), 2) for p in principles}
        print("\n=== BATCH MEAN (the honing dashboard) ===")
        print("  " + "  ".join(f"{p}={batch[p]:.2f}" for p in principles))
        weakest = min(batch, key=batch.get)
        names = {"C": "Comparison classes", "O": "Outside view", "H": "Hunt", "A": "Adjust often",
                 "M": "Make precise", "P": "Pre-mortem", "S": "Synthesize", "N": "No sacred cows"}
        print(f"  WEAKEST PRINCIPLE -> {weakest} ({names[weakest]}) at {batch[weakest]:.2f}  ← hone this next")
        if args.out:
            json.dump({"runs": cards, "batch_mean": batch}, open(args.out, "w"), indent=2, ensure_ascii=False)
            print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
