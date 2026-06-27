#!/usr/bin/env python3
"""Approach 2: tabular neural network on the shared feature set."""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_common import (
    build_training_data,
    chronological_holdout_indices,
    load_inputs,
    make_bundle,
    run_walk_forward_backtest,
    sample_weight_array,
)


EXPERIMENT_NAME = "Neural Network + Dixon-Coles"


class TorchTabularClassifier:
    """Small PyTorch MLP with sklearn-like ``predict_proba``."""

    def __init__(self, epochs: int = 80, batch_size: int = 512, lr: float = 1e-3, seed: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y, sample_weight=None, val_data=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X_np = self.imputer.fit_transform(X)
        X_np = self.scaler.fit_transform(X_np).astype(np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        w_np = np.asarray(sample_weight if sample_weight is not None else np.ones(len(y_np)), dtype=np.float32)

        n_features = X_np.shape[1]

        class Net(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(width, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 3),
                )

            def forward(self, x):
                return self.net(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net(n_features).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        X_t = torch.from_numpy(X_np).to(device)
        y_t = torch.from_numpy(y_np).to(device)
        w_t = torch.from_numpy(w_np).to(device)
        best_state = None
        best_val = float("inf")
        stale = 0
        rng = np.random.default_rng(self.seed)

        if val_data is not None:
            X_val, y_val = val_data
            Xv_np = self.scaler.transform(self.imputer.transform(X_val)).astype(np.float32)
            Xv_t = torch.from_numpy(Xv_np).to(device)
            yv_t = torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(device)
        else:
            Xv_t = yv_t = None

        for _epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(y_np))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = X_t[idx]
                yb = y_t[idx]
                wb = w_t[idx]
                opt.zero_grad()
                loss_each = F.cross_entropy(self.model(xb), yb, reduction="none")
                loss = (loss_each * wb).mean()
                loss.backward()
                opt.step()

            if Xv_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(F.cross_entropy(self.model(Xv_t), yv_t).item())
                if val_loss + 1e-5 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= 10:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F

        if self.model is None:
            raise RuntimeError("TorchTabularClassifier is not fitted")
        X_np = self.scaler.transform(self.imputer.transform(X)).astype(np.float32)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_np).to(device))
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def train_bundle():
    results_df, country_history, odds, squad_values = load_inputs()
    training = build_training_data(results_df, country_history, odds, squad_values)
    weights = sample_weight_array(training)
    train_idx, val_idx = chronological_holdout_indices(training.dates)
    notes: list[str] = []

    try:
        import torch  # noqa: F401
    except Exception as exc:
        notes.append(f"PyTorch unavailable ({exc}); used sklearn MLPClassifier fallback")
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128, 64),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=512,
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=int(os.getenv("SKLEARN_MLP_MAX_ITER", "80")),
                        random_state=42,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(training.X, training.y)
    else:
        model = TorchTabularClassifier(epochs=int(os.getenv("TORCH_MLP_EPOCHS", "80")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                training.X.iloc[train_idx],
                training.y[train_idx],
                sample_weight=weights[train_idx],
                val_data=(training.X.iloc[val_idx], training.y[val_idx]),
            )
        notes.append("PyTorch MLP architecture: 52-ish -> 256 -> 128 -> 64 -> 3 with BatchNorm/Dropout")

    return make_bundle(
        model,
        training,
        results_df=results_df,
        country_history=country_history,
        odds=odds,
        squad_values=squad_values,
        tune_poisson_blend=True,
        notes=notes,
    )


def main():
    bundle = train_bundle()
    run_walk_forward_backtest(EXPERIMENT_NAME, bundle)


if __name__ == "__main__":
    main()
