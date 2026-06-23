"""
models/ensemble.py
------------------
The Stage 2 "second model": a calibrated RF + XGBoost ensemble that predicts
P(home win) from the Elo/Pythag/rest features. Trained on a time-ordered split,
isotonic-calibrated, and significance-tested against the Elo baseline so it only
"ships" if it clears paired-bootstrap p<0.05 on held-out games.

This is the rig that answers the video's question: is the new model actually
smarter, or just lucky?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss

from models.features import FEATURE_COLS
from backtest.walk_forward import paired_bootstrap_pvalue

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    _HAVE_XGB = False


class EnsembleModel:
    def __init__(self):
        self.rf = None
        self.gb = None

    def fit(self, X, y):
        rf = RandomForestClassifier(n_estimators=300, max_depth=5,
                                    min_samples_leaf=20, random_state=0)
        if _HAVE_XGB:
            gb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                               subsample=0.8, eval_metric="logloss",
                               random_state=0)
        else:
            gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=0)
        # isotonic calibration via internal CV on the training split
        self.rf = CalibratedClassifierCV(rf, method="isotonic", cv=3).fit(X, y)
        self.gb = CalibratedClassifierCV(gb, method="isotonic", cv=3).fit(X, y)
        return self

    def predict_proba(self, X):
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_gb = self.gb.predict_proba(X)[:, 1]
        return 0.5 * (p_rf + p_gb)


def time_split(df: pd.DataFrame, test_frac: float = 0.30):
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _per_game_logloss(p, y, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_and_gate(df: pd.DataFrame, test_frac: float = 0.30,
                   alpha: float = 0.05) -> dict:
    """
    Train the ensemble on the early games, test on the most recent games, and
    decide ship/hold vs the Elo baseline via paired bootstrap on log-loss.
    """
    train, test = time_split(df, test_frac)
    Xtr, ytr = train[FEATURE_COLS].values, train["home_win"].values
    Xte, yte = test[FEATURE_COLS].values, test["home_win"].values

    model = EnsembleModel().fit(Xtr, ytr)
    p_model = model.predict_proba(Xte)
    p_elo = test["elo_p"].values            # incumbent baseline on same games

    loss_model = _per_game_logloss(p_model, yte)
    loss_elo = _per_game_logloss(p_elo, yte)
    p_value = paired_bootstrap_pvalue(loss_model, loss_elo)

    return {
        "n_train": len(train), "n_test": len(test),
        "model_logloss": float(loss_model.mean()),
        "elo_logloss": float(loss_elo.mean()),
        "model_brier": float(brier_score_loss(yte, p_model)),
        "elo_brier": float(brier_score_loss(yte, np.clip(p_elo, 1e-6, 1-1e-6))),
        "model_acc": float(np.mean((p_model > 0.5) == (yte == 1))),
        "p_value_vs_elo": p_value,
        "ship": bool(p_value < alpha),
        "xgboost": _HAVE_XGB,
        "_model": model,
    }


def save_model(model: EnsembleModel, season: int):
    import joblib
    out_dir = __import__("config").ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ensemble_{season}.joblib"
    joblib.dump(model, path)
    return path


def load_model(season: int):
    import joblib, config
    path = config.ARTIFACTS / f"ensemble_{season}.joblib"
    return joblib.load(path) if path.exists() else None


if __name__ == "__main__":
    import argparse, datetime as dt
    from models.features import build_training_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    args = ap.parse_args()
    df = build_training_table(args.season)
    r = train_and_gate(df)
    print(f"train {r['n_train']}  test {r['n_test']}  (xgboost={r['xgboost']})")
    print(f"ensemble  log-loss {r['model_logloss']:.4f}  brier {r['model_brier']:.4f}  acc {r['model_acc']:.3f}")
    print(f"elo base  log-loss {r['elo_logloss']:.4f}  brier {r['elo_brier']:.4f}")
    print(f"paired-bootstrap p vs Elo: {r['p_value_vs_elo']:.4f}")
    if r["ship"]:
        path = save_model(r["_model"], args.season)
        print(f"SHIP - ensemble beats Elo at p<0.05. Saved -> {path.name}")
    else:
        print("HOLD - ensemble does NOT beat Elo at p<0.05. Not shipping "
              "(this is the gate working, not a failure).")
