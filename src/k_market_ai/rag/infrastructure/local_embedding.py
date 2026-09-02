from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_ID = f"{MODEL_NAME}@{MODEL_REVISION}"
EMBEDDING_DIMENSIONS = 384


class LocalEmbeddingAdapter:
    def __init__(self) -> None:
        self._encoder: SentenceTransformer | None = None
        self._load_lock = threading.Lock()

    @property
    def model(self) -> str:
        return MODEL_ID

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIMENSIONS

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load_encoder)

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, tuple(texts))

    def _encode(self, texts: tuple[str, ...]) -> list[tuple[float, ...]]:
        encoder = self._load_encoder()
        encoded = encoder.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = cast(NDArray[np.float32], encoded)
        return [tuple(float(value) for value in vector) for vector in vectors]

    def _load_encoder(self) -> SentenceTransformer:
        if self._encoder is not None:
            return self._encoder
        with self._load_lock:
            if self._encoder is not None:
                return self._encoder
            import torch
            from sentence_transformers import SentenceTransformer

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            self._encoder = SentenceTransformer(
                MODEL_NAME,
                revision=MODEL_REVISION,
                device=device,
            )
        return self._encoder
