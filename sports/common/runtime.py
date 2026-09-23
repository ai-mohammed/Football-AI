"""One hardware policy for the CLI, Streamlit and model loaders."""
import warnings

import torch


def resolve_device(requested="auto"):
    requested = str(requested).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested.startswith("cuda"):
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            warnings.warn("GPU CUDA indisponible sur cet hôte : utilisation du CPU.", RuntimeWarning)
            return "cpu"
    elif requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            warnings.warn("GPU MPS indisponible : utilisation du CPU.", RuntimeWarning)
            return "cpu"
    elif requested != "cpu":
        raise ValueError(f"Unsupported device: {requested}")
    return requested
