from typing import Generator, Iterable, List, TypeVar

import numpy as np
import cv2
import supervision as sv
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from sports.common.runtime import resolve_device

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    Jersey-colour clustering with an optional SigLIP embedding backend.
    Label 2 means unknown. Team A/B are arbitrary cluster names per video.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32, method: str = 'jersey'):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = resolve_device(device)
        self.batch_size = batch_size
        if method not in ('jersey', 'siglip'):
            raise ValueError('Team method must be jersey or siglip')
        self.method = method
        if method == 'siglip':
            from transformers import AutoImageProcessor, SiglipVisionModel
            self.features_model = SiglipVisionModel.from_pretrained(
                SIGLIP_MODEL_PATH).to(self.device).eval()
            self.processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_PATH, use_fast=False)
        self.reducer = None
        self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.fitted = False

    @staticmethod
    def jersey_features(crops):
        """Torso chromaticity; reduce luminance influence to tolerate shadows.

        No grass-colour mask: a football shirt may itself be green.
        This is a colour baseline, not a learned team/identity recognizer.
        """
        features = []
        for crop in crops:
            h, w = crop.shape[:2]
            torso = crop[int(h * .15):max(int(h * .5), int(h * .15) + 1),
                         int(w * .2):max(int(w * .8), int(w * .2) + 1)]
            if not torso.size:
                features.append([0, 0, 0])
                continue
            lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
            features.append(np.median(lab.reshape(-1, 3), axis=0) / 255 * [0.25, 1, 1])
        return np.asarray(features, dtype=np.float32).reshape(-1, 3)

    def extract_features(
        self, crops: List[np.ndarray], verbose: bool = True
    ) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.
            verbose (bool): Whether to show a progress bar. Useful to disable
                when calling this once per video frame (e.g. from `predict`),
                where a per-batch progress bar is just noise.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        if verbose:
            batches = tqdm(batches, desc='Embedding extraction')
        data = []
        with torch.inference_mode():
            for batch in batches:
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        self.fitted = False
        if len(crops) < 2:
            return
        if self.method == 'jersey':
            projections = self.jersey_features(crops)
        else:
            data = self.extract_features(crops)
            self.reducer = PCA(n_components=min(16, len(data), data.shape[1]), whiten=True, random_state=42)
            projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        self.fitted = bool(np.linalg.norm(np.diff(self.cluster_model.cluster_centers_, axis=0)) >= 0.04)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.empty(0, dtype=int)
        if not self.fitted:
            return np.full(len(crops), 2, dtype=int)

        if self.method == 'jersey':
            projections = self.jersey_features(crops)
        else:
            data = self.extract_features(crops, verbose=False)
            projections = self.reducer.transform(data)
        distances = self.cluster_model.transform(projections)
        predictions = distances.argmin(axis=1)
        margin = (distances.max(axis=1) - distances.min(axis=1)) / np.maximum(distances.sum(axis=1), 1e-8)
        predictions[margin < 0.15] = 2  # Unknown, do not force an ambiguous assignment.
        return predictions
