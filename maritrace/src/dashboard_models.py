"""Saved-model accessors used by the Streamlit frontend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_satellite_model(checkpoint_path: str | Path) -> Any:
    """Load the repository MobileNetV2 architecture from a saved checkpoint."""
    import torch
    from .satellite_classifier import MobileNetV2SAR

    model = MobileNetV2SAR(pretrained=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_satellite_score(model: Any, image: Image.Image) -> float:
    """Return the saved model's sigmoid score after its documented preprocessing."""
    import torch
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        return float(model.predict_proba(preprocess(image.convert("RGB")).unsqueeze(0)).item())

