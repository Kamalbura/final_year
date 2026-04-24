from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data.dataset import build_datasets
from src.evaluation.metrics import mae, r2, rmse
from src.models.transformers import RTTransformerForecaster, TransformerForecaster
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer attention models for AQ forecasting.")
    parser.add_argument("--model", choices=["transformer", "rt_transformer"], default="transformer")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_model(model_name: str, input_dim: int, horizon: int, output_dim: int, model_cfg: dict) -> nn.Module:
    common_kwargs = dict(
        input_dim=input_dim,
        model_dim=int(model_cfg["model_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        horizon=horizon,
        output_dim=output_dim,
    )
    if model_name == "rt_transformer":
        return RTTransformerForecaster(local_window=int(model_cfg["local_window"]), **common_kwargs)
    return TransformerForecaster(**common_kwargs)


def _flatten_targets(y: np.ndarray) -> np.ndarray:
    return y.reshape(y.shape[0], -1)


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    set_global_seed(int(config["project"]["seed"]))
    datasets = build_datasets(config)
    (x_train, y_train) = datasets["train"]
    (x_val, y_val) = datasets["val"]
    input_dim = x_train.shape[-1]
    horizon = y_train.shape[1]
    output_dim = y_train.shape[2]
    training_cfg = config["training"]
    model_cfg = config["models"]["attention"][args.model]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(_flatten_targets(y_train), dtype=torch.float32)),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(_flatten_targets(y_val), dtype=torch.float32)),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
    )

    model = _build_model(
        args.model,
        input_dim=input_dim,
        horizon=horizon,
        output_dim=output_dim,
        model_cfg=model_cfg,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    criterion = nn.MSELoss()

    for epoch in range(int(training_cfg["max_epochs"])):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred.reshape(pred.size(0), -1), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                flat_pred = pred.reshape(pred.size(0), -1)
                loss = criterion(flat_pred, y_batch)
                val_losses.append(loss.item())
                all_preds.append(flat_pred.numpy())
                all_targets.append(y_batch.numpy())

        val_pred = np.concatenate(all_preds, axis=0)
        val_target = np.concatenate(all_targets, axis=0)
        scaler = datasets["scaler"]
        if scaler is not None:
            val_pred = scaler.inverse_transform(val_pred.reshape(-1, output_dim)).reshape(-1, horizon, output_dim)
            val_target = scaler.inverse_transform(val_target.reshape(-1, output_dim)).reshape(-1, horizon, output_dim)
        print(
            f"epoch={epoch + 1} train_loss={float(np.mean(train_losses)):.4f} "
            f"val_loss={float(np.mean(val_losses)):.4f} rmse={rmse(val_target, val_pred):.4f} "
            f"mae={mae(val_target, val_pred):.4f} r2={r2(val_target, val_pred):.4f}"
        )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{args.model}.pth")
    if datasets["scaler"] is not None:
        joblib.dump(datasets["scaler"], output_dir / f"{args.model}_scaler.pkl")
    (output_dir / f"{args.model}_summary.json").write_text(
        json.dumps({"model": args.model, "input_dim": input_dim, "horizon": horizon, "output_dim": output_dim}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
