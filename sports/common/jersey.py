"""Optional soccer-specific jersey recognition with explicit unreadable results.

The external ViT-S checkpoint is by Lukasz Grad (CVPRW 2025).
Its upstream LICENSE says CC-BY-NC-SA-4.0; see the provenance note below.
See docs/JERSEY_NUMBERS.md for provenance, setup and the recognition limits.
This adapter uses timm features and evaluates the checkpoint's digit head.
"""
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'examples/soccer/data/jersey-small16.safetensors'


def model_path():
    return Path(os.environ.get('FOOTBALL_JERSEY_MODEL', str(DEFAULT_MODEL)))


def specialist_available():
    return model_path().is_file() and all(importlib.util.find_spec(name) is not None
                                        for name in ('timm', 'safetensors'))


def torso_crop(crop):
    h = crop.shape[0]
    return crop[h//6:h-h//3]


class JerseyRecognizer:
    """Read a batch of full player crops; scores are model evidence, not accuracy."""
    def __init__(self, device='cpu'):
        import timm
        import torch
        from safetensors.torch import load_file
        self.device = device
        weights = load_file(str(model_path()))
        self.model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=False, num_classes=0)
        backbone = {k.removeprefix('backbone.'): v for k, v in weights.items() if k.startswith('backbone.')}
        self.model.load_state_dict(backbone, strict=True)
        self.model.to(device).eval()
        self.head = {k: v.to(device) for k, v in weights.items() if k.startswith('classifier.')}
        self.torch = torch

    def read(self, crops):
        if not crops:
            return []
        torch = self.torch
        images = [cv2.resize(cv2.cvtColor(torso_crop(c), cv2.COLOR_BGR2RGB), (224, 224),
                             interpolation=cv2.INTER_CUBIC) for c in crops]
        batch = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2)).float().to(self.device)/127.5-1.
        with torch.inference_mode():
            features = self.model.forward_features(batch)[:, 0]
            positioned = features[:, None, :]*self.head['classifier.position_embeddings'][None]
            digit_logits = torch.nn.functional.linear(positioned, self.head['classifier.digit_classifier.weight'])
            digit_logits += self.head['classifier.position_biases'][None]
            numbers = torch.arange(10, 100, device=self.device)
            logits = torch.cat((digit_logits[:, 0], digit_logits[:, 1, numbers//10]
                                + digit_logits[:, 2, numbers%10]), dim=1)
            # log(alpha) = log(1+exp(logit)); stable even for a confident model.
            log_alpha = torch.nn.functional.softplus(logits)
            log_total = torch.logsumexp(log_alpha, dim=1)
            probs = (log_alpha-log_total[:, None]).exp()
            uncertainty = (np.log(100)-log_total).exp().cpu().numpy()
            scores, labels = probs.max(dim=1)
        return [{'number': str(int(n)) if 1 <= int(n) <= 99 else None,
                 'confidence': float(c), 'uncertainty': float(u), 'source': 'soccer_vit_small16'}
                for n, c, u in zip(labels.cpu(), scores.cpu(), uncertainty)]


def crop_quality(crop):
    """Native crop suitability, before enlargement; this cannot create detail."""
    if crop is None or not crop.size or crop.shape[0] < 40 or crop.shape[1] < 14:
        return 0.
    torso = torso_crop(crop)
    sharpness = cv2.Laplacian(cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    return float(min(1., sharpness/120) * min(1., crop.shape[0]/90))
