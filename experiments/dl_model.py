"""EXPERIMENT (not used by the app): small PyTorch MLP on the same split as the classical models.

    python -m experiments.dl_model

Purpose: check honestly whether a neural network helps on this small tabular
dataset. It uses the *same* grouped hold-out split, a preprocessor fit on the
training partition only, early stopping on a validation slice carved out of the
training partition, and the same metrics code. Results are written to
experiments/dl_results.json and are reported in docs/MODEL_CARD.md as they are.
"""
# torch must be imported before scikit-learn on some Windows installs (DLL load order).
import torch  # isort: skip
import torch.nn as nn  # isort: skip

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.metrics import compute_metrics
from src.ml_model import holdout_split
from src.preprocessing import FEATURES, ID_COLUMN, TARGET, build_preprocessor, load_data


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)   # logits; sigmoid applied at prediction time


def run(seed: int = config.SEED, max_epochs: int = 300, patience: int = 20) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data().reset_index(drop=True)
    tr_idx, te_idx = holdout_split(df, seed)
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    # validation slice (grouped) for early stopping; the test set is never used for training decisions
    fit_i, val_i = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
        train[FEATURES], train[TARGET], train[ID_COLUMN]))
    prep = build_preprocessor().fit(train.iloc[fit_i][FEATURES])

    def tensors(frame):
        return (torch.tensor(prep.transform(frame[FEATURES]), dtype=torch.float32, device=device),
                torch.tensor(frame[TARGET].to_numpy(), dtype=torch.float32, device=device))
    Xf, yf = tensors(train.iloc[fit_i])
    Xv, yv = tensors(train.iloc[val_i])
    Xt, _ = tensors(test)

    model = MLP(len(FEATURES)).to(device)
    pos_weight = (yf == 0).sum() / (yf == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    best, best_state, bad = float("inf"), None, 0
    for _epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(Xf), yf).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(Xt)).cpu().numpy()
    metrics = compute_metrics(test[TARGET].to_numpy(), score)
    return {"seed": seed, "device": str(device), "val_loss": best, "test_metrics": metrics}


if __name__ == "__main__":
    result = run()
    out = Path(__file__).with_name("dl_results.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    m = result["test_metrics"]
    print(f"MLP test: ROC-AUC={m['roc_auc']:.3f} sensitivity={m['recall_sensitivity']:.3f} "
          f"specificity={m['specificity']:.3f} precision={m['precision']} (device={result['device']})")
