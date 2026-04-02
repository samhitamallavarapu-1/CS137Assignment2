from pathlib import Path
import torch
import pandas as pd

DATASET_ROOT = Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset")


# -------------------------------
# DATA LOADING
# -------------------------------

def load_metadata():
    return torch.load(DATASET_ROOT / "metadata.pt", weights_only=False)


def load_targets():
    return torch.load(DATASET_ROOT / "targets.pt", weights_only=False)


def load_input(times, t_idx):
    dt = pd.Timestamp(times[t_idx])
    path = DATASET_ROOT / "inputs" / str(dt.year) / f"X_{dt.strftime('%Y%m%d%H')}.pt"
    x = torch.load(path, weights_only=True).float()
    return x, dt, path


# -------------------------------
# NORMALIZATION
# -------------------------------

def normalize(x, mean, std):
    return (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def unnormalize(x, mean, std):
    return x * std.view(1, 1, -1) + mean.view(1, 1, -1)


# -------------------------------
# METRICS HELPERS
# -------------------------------

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def rmse_per_channel(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=0))


def apcp_metrics(preds, tgts, binary, threshold=2.0, apcp_idx=5):
    rain_mask = tgts[:, apcp_idx] > threshold

    if rain_mask.sum() > 0:
        rmse_rain = torch.sqrt(
            torch.mean((preds[rain_mask, apcp_idx] - tgts[rain_mask, apcp_idx]) ** 2)
        ).item()
    else:
        rmse_rain = float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(binary.numpy(), preds[:, apcp_idx].numpy()))
    except Exception:
        auc = float("nan")

    return rmse_rain, auc


# -------------------------------
# SALIENCY HELPERS
# -------------------------------

def compute_saliency(model, x, target_idx):
    """
    x: (1, H, W, C)
    """
    x = x.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    output = model(x)

    target = output[0, target_idx]
    target.backward()

    saliency = x.grad[0].detach().cpu()
    return saliency, output.detach().cpu()[0]


def spatial_saliency(saliency):
    return saliency.abs().mean(dim=-1)


def channel_saliency(saliency):
    return saliency.abs().mean(dim=(0, 1))


# -------------------------------
# VISUALIZATION HELPERS
# -------------------------------

def normalize_map(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)