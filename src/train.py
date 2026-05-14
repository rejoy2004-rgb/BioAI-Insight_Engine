import os
import sys
import argparse
import joblib

# Make sure `src/` is on the Python path when run directly 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import preprocess
from src.model_comparison import train_and_compare, get_best_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BioAI models")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases experiment tracking",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data to hold out for testing (default: 0.20)",
    )
    return parser.parse_args()


def save_best_model(model, model_name: str) -> None:
    """
    Persists the best model to disk using joblib.

    WHY JOBLIB OVER PICKLE:
      joblib is optimised for objects that contain large numpy arrays (like
      RandomForest's estimators) — it compresses them efficiently and is the
      scikit-learn recommended serialisation format.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "model_name": model_name}, MODEL_PATH)
    print(f"[Train] Best model ({model_name}) saved → {MODEL_PATH}")


def log_to_wandb(model_name: str, metrics: dict) -> None:
    """
    Logs a single training run to Weights & Biases.

    W&B tracks every run (hyperparameters, metrics, system info) so you can
    compare experiments over time and avoid the 'which notebook was that?' problem.

    Skips gracefully if wandb is not installed — keeps the script usable without it.
    """
    try:
        import wandb
        run = wandb.init(
            project="BioAI-Insight-Engine",
            name=model_name,
            tags=["cancer-prediction", "classification"],
            reinit=True,
        )
        wandb.log(
            {
                "model_name": model_name,
                **metrics,
            }
        )
        run.finish()
        print(f"[Train] W&B run logged for '{model_name}'")
    except ImportError:
        print("[Train] wandb not installed — skipping experiment tracking.")
        print("        Install with: pip install wandb")


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("  BioAI Insight Engine — Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocess (includes train/test split and scaler fitting)
    print("\n[Train] Step 1/3 — Preprocessing data ...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        test_size=args.test_size
    )

    # Step 2: Train all candidate models and compare
    print("\n[Train] Step 2/3 — Training and comparing models ...")
    results, fitted_models = train_and_compare(
        X_train, X_test, y_train, y_test, save_plots=True
    )

    # Step 3: Select and save the best model
    print("\n[Train] Step 3/3 — Selecting best model ...")
    best_name = get_best_model(results)
    best_model = fitted_models[best_name]
    save_best_model(best_model, best_name)

    # Print final summary table
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"  {'Model':<25} {'Acc':>6} {'Prec':>7} {'Rec':>6} {'F1':>6} {'AUC':>7}")
    print("  " + "-" * 56)
    for model_name, metrics in results.items():
        flag = " ← BEST" if model_name == best_name else ""
        print(
            f"  {model_name:<25} "
            f"{metrics['Accuracy']:>6.3f} "
            f"{metrics['Precision']:>7.3f} "
            f"{metrics['Recall']:>6.3f} "
            f"{metrics['F1-Score']:>6.3f} "
            f"{metrics['ROC-AUC']:>7.3f}{flag}"
        )
    print("=" * 60)

    # Optional W&B logging
    if args.wandb:
        print("\n[Train] Logging all runs to Weights & Biases ...")
        for model_name, metrics in results.items():
            log_to_wandb(model_name, metrics)

    print("\n[Train] Training complete. All outputs saved.")


if __name__ == "__main__":
    main()
