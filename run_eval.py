"""Run the fixed CHAMPS KNOW eval set on a provider and save full runs to a
labelled directory, so successive prompt versions can be scored and compared.

  uv run run_eval.py --provider deepseek --trials 1 --label baseline
  uv run score_champs.py eval_runs/baseline

No market prior is injected (prior=None) on purpose: these aren't markets, so the
agent must form its own comparison_class/base_rate — which is exactly the
outside-view discipline we're honing. Runs save to eval_runs/<label>/ (kept out of
runs/ so the calibration set isn't polluted by repeated honing passes).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import forecaster

EVAL_PATH = Path(__file__).resolve().parent / "eval" / "champs_eval.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the CHAMPS KNOW eval set")
    ap.add_argument("--provider", choices=sorted(forecaster.PROVIDERS), default="deepseek")
    ap.add_argument("--trials", type=int, default=1, help="trials per question (1 for cheap honing)")
    ap.add_argument("--label", required=True, help="output dir label, e.g. 'baseline' or 'v2-grounding'")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    forecaster.set_provider(args.provider)
    out_dir = Path("eval_runs") / args.label
    out_dir.mkdir(parents=True, exist_ok=True)  # save_run's mkdir won't create parents
    forecaster.RUNS_DIR = out_dir  # redirect save_run away from the real runs/ dir

    questions = [q["question"] for q in json.load(open(EVAL_PATH))]
    logging.info("Eval set: %d questions | provider=%s trials=%d -> %s",
                 len(questions), args.provider, args.trials, out_dir)

    for i, q in enumerate(questions, 1):
        logging.info("[%d/%d] %s", i, len(questions), q)
        t0 = time.perf_counter()
        try:
            res = forecaster.aggregate_forecasts(q, prior=None, num_trials=args.trials)
            forecaster.save_run(q, res, prior=None, num_trials=args.trials)
            logging.info("  -> p=%.3f (%.0fs)", res.probability, time.perf_counter() - t0)
        except Exception as exc:
            logging.warning("  FAILED: %s", exc)

    logging.info("DONE -> score with: uv run score_champs.py %s", out_dir)


if __name__ == "__main__":
    main()
