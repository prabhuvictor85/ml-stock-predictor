"""
inspect_hpo.py — Read and visualise the Optuna HPO study at any point.

Works mid-run (partial results) or after completion.
Saves all plots as HTML files you can open in any browser.

Usage:
    python inspect_hpo.py
    python inspect_hpo.py --db artefacts/nse_local/checkpoints/optuna_hpo.db
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

DB_DEFAULT = Path("artefacts/nse_local/checkpoints/optuna_hpo.db")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DB_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    db_path = args.db

    if not db_path.exists():
        print(f"Optuna DB not found: {db_path}")
        print("The run is still in feature engineering / hasn't started HPO yet.")
        print("Re-run this script once HPO has started.")
        sys.exit(0)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{db_path.as_posix()}"

    # ── List all studies in the DB ─────────────────────────────────────────
    study_names = optuna.get_all_study_names(storage)
    print(f"Studies in DB: {study_names}")
    if not study_names:
        print("No studies found yet — HPO hasn't written any trials.")
        sys.exit(0)

    study_name = study_names[0]          # there's only one: "nse_local_hpo"
    study = optuna.load_study(study_name=study_name, storage=storage)

    # ── Summary ────────────────────────────────────────────────────────────
    trials      = study.trials
    completed   = [t for t in trials if t.state.name == "COMPLETE"]
    pruned      = [t for t in trials if t.state.name == "PRUNED"]
    failed      = [t for t in trials if t.state.name == "FAIL"]

    print(f"\n{'='*55}")
    print(f"Study      : {study_name}")
    print(f"DB         : {db_path}")
    print(f"{'='*55}")
    print(f"Total trials  : {len(trials)}")
    print(f"  Completed   : {len(completed)}")
    print(f"  Pruned      : {len(pruned)}")
    print(f"  Failed      : {len(failed)}")

    if not completed:
        print("\nNo completed trials yet — HPO is still running.")
        sys.exit(0)

    best = study.best_trial
    print(f"\nBest trial    : #{best.number}")
    print(f"Best NDCG obj : {best.value:.5f}")
    print(f"\nBest params:")
    for k, v in best.params.items():
        print(f"  {k:<25} = {v}")

    # Top-5 trials
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\nTop-5 trials:")
    print(f"  {'#':<5} {'NDCG obj':<12} {'num_leaves':<12} {'lr':<10} {'n_est':<8} {'top_K'}")
    print(f"  {'-'*60}")
    for t in top5:
        p = t.params
        print(f"  {t.number:<5} {t.value:<12.5f} "
              f"{p.get('num_leaves','?'):<12} "
              f"{p.get('lr', p.get('learning_rate','?')):<10.4f} "
              f"{p.get('n_estimators','?'):<8} "
              f"{p.get('feature_top_K','?')}")

    # ── Save plots ─────────────────────────────────────────────────────────
    out_dir = Path("reports/hpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import plotly  # noqa — just check it's installed

        plots = {
            "optimization_history" : optuna.visualization.plot_optimization_history(study),
            "param_importances"    : optuna.visualization.plot_param_importances(study),
            "parallel_coordinate"  : optuna.visualization.plot_parallel_coordinate(study),
            "slice"                : optuna.visualization.plot_slice(study),
        }

        # contour for the two most important params (need >= 2 completed trials)
        if len(completed) >= 5:
            plots["contour_lr_leaves"] = optuna.visualization.plot_contour(
                study, params=["num_leaves", "lr"]
            )

        for name, fig in plots.items():
            path = out_dir / f"{name}.html"
            fig.write_html(str(path))
            print(f"  Saved: {path}")

        print(f"\nAll plots saved to {out_dir}/")
        print("Open any .html file in your browser to view interactively.")

    except ImportError:
        print("\nplotly not installed — skipping HTML plots.")
        print("Install with: pip install plotly kaleido")

    # ── Plain-text convergence curve ───────────────────────────────────────
    print(f"\nConvergence (every 10 trials):")
    best_so_far = float("-inf")
    for t in completed:
        if t.value > best_so_far:
            best_so_far = t.value
        if t.number % 10 == 0 or t.number == completed[-1].number:
            bar = "#" * int(max(0, best_so_far) * 40)
            print(f"  Trial {t.number:>4}: best={best_so_far:.5f}  {bar}")


if __name__ == "__main__":
    main()

