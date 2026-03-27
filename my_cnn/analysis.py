from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import numpy as np


def compute_input_saliency(
    model: torch.nn.Module,
    weights_path: Union[str, Path],
    input_tensor: torch.Tensor,
    target_index: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    apply_abs: bool = True,
    reduce_batch: bool = True,
) -> Dict[str, Any]:
    """
    Compute gradient-based saliency for a PyTorch model input.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated model architecture.
    weights_path : str or Path
        Path to .pth weights file (state_dict).
    input_tensor : torch.Tensor
        Input tensor.
        Expected shape for your weather CNN: (B, H, W, C)
        but this function works for any differentiable input shape.
    target_index : int or None
        Which output neuron to explain.
        If None:
          - if model output is scalar, uses that scalar
          - if model output is vector, uses argmax for each sample
    device : str, torch.device, or None
        Device to run on. If None, uses CUDA if available else CPU.
    apply_abs : bool
        If True, returns absolute gradient saliency.
    reduce_batch : bool
        If True and batch size > 1, also returns batch-averaged summaries.

    Returns
    -------
    result : dict
        {
            "saliency": full saliency tensor, same shape as input_tensor,
            "feature_scores": per-sample per-feature saliency,
            "feature_scores_mean": batch-mean per-feature saliency (if reduce_batch),
            "target_index_used": target index/indices used,
            "output": model output tensor (detached CPU),
        }

    Notes
    -----
    - Saliency is d(output[target])/d(input).
    - For input shape (B, H, W, C), feature_scores has shape (B, C),
      computed by averaging saliency across H and W.
    """
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = model.to(device)
    model.eval()

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    x = input_tensor.detach().clone().to(device)
    x.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    output = model(x)

    if output.ndim == 1:
        # shape: (B,)
        scalar_output = output.sum()
        target_used = "scalar_output"
    elif output.ndim == 2:
        # shape: (B, num_outputs)
        if target_index is None:
            chosen_idx = output.argmax(dim=1)
        else:
            if not (0 <= target_index < output.shape[1]):
                raise ValueError(f"target_index={target_index} out of range for output shape {tuple(output.shape)}")
            chosen_idx = torch.full((output.shape[0],), target_index, device=device, dtype=torch.long)

        scalar_output = output.gather(1, chosen_idx.unsqueeze(1)).sum()
        target_used = chosen_idx.detach().cpu()
    else:
        raise ValueError(
            f"Unsupported model output shape {tuple(output.shape)}. "
            "Expected scalar/vector outputs shaped (B,) or (B, K)."
        )

    scalar_output.backward()

    saliency = x.grad.detach()
    if apply_abs:
        saliency = saliency.abs()

    saliency_cpu = saliency.cpu()
    output_cpu = output.detach().cpu()

    result: Dict[str, Any] = {
        "saliency": saliency_cpu,
        "target_index_used": target_used,
        "output": output_cpu,
    }

    # For your weather model input shape (B, H, W, C), summarize per feature/channel
    if saliency_cpu.ndim == 4:
        # assume channels-last: (B, H, W, C)
        feature_scores = saliency_cpu.mean(dim=(1, 2))  # (B, C)
        result["feature_scores"] = feature_scores
        if reduce_batch:
            result["feature_scores_mean"] = feature_scores.mean(dim=0)  # (C,)
    elif saliency_cpu.ndim == 2:
        # shape (B, C)
        feature_scores = saliency_cpu
        result["feature_scores"] = feature_scores
        if reduce_batch:
            result["feature_scores_mean"] = feature_scores.mean(dim=0)
    else:
        # Generic fallback: flatten all but batch dimension
        feature_scores = saliency_cpu.reshape(saliency_cpu.shape[0], -1)
        result["feature_scores"] = feature_scores
        if reduce_batch:
            result["feature_scores_mean"] = feature_scores.mean(dim=0)

    return result

def normalize_saliency_map(saliency: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize saliency to [0, 1] per sample.
    Expects shape (B, H, W, C) or similar.
    """
    s = saliency.clone().float()
    flat = s.view(s.shape[0], -1)
    mins = flat.min(dim=1).values.view(-1, *([1] * (s.ndim - 1)))
    maxs = flat.max(dim=1).values.view(-1, *([1] * (s.ndim - 1)))
    return (s - mins) / (maxs - mins + eps)

if __name__ == "__main__":
    pass