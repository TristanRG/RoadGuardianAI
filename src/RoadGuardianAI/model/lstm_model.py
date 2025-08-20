import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]          
        logits = self.head(last).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs


def compute_metrics_np(y_true: np.ndarray, y_prob: np.ndarray):
    if len(np.unique(y_true)) == 1:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    return {"auc": auc, "accuracy": acc}


def train_lstm_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str = "cpu",
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
):
    device = torch.device(device)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_auc = -np.inf
    wait = 0
    last_val_metrics = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            opt.zero_grad()
            logits, probs = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
        train_loss = total_loss / max(1, total_samples)

        model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                logits, probs = model(xb)
                ys.append(yb.numpy())
                ps.append(probs.cpu().numpy())
        if len(ys) == 0:
            break
        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        val_metrics = compute_metrics_np(ys, ps)
        last_val_metrics = val_metrics

        if not np.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, last_val_metrics


def save_torch_model(path: str, model: nn.Module, metadata: Dict[str, Any] = None):
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_torch_model(path: str, model_cls, device: str = "cpu", **model_kwargs):
    payload = torch.load(path, map_location=device)
    md = payload.get("metadata", {})
    input_dim = md.get("input_dim", model_kwargs.get("input_dim"))
    model = model_cls(input_dim=input_dim, **{k: v for k, v in model_kwargs.items() if k != "input_dim"})
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
