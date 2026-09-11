from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np


class EmbeddingModel:
    """Lazy-loaded local embedding model so application startup stays fast."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 4):
        self.model_name = model_name
        self.requested_device = device
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _choose_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except (ImportError, AttributeError):
            return "cpu"
        return "cpu"

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self._choose_device(),
                    )
        return self._model

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 20,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
