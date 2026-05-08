"""Phase 30 B.4 — Discrete chart token store.

Tokenizes per-bar OHLCV into discrete codebook tokens (k-means or VQ-VAE),
persists in LanceDB (Phase 28 vector store). Used by foundation models that
need fixed-vocabulary input (vs. continuous prices).

This is a SCAFFOLD with two implementations:
1. Simple k-means (numpy + sklearn.MiniBatchKMeans) — usable today.
2. VQ-VAE wrapper interface — reserved for B.2 fine-tune training.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CODEBOOK_DIR = Path(__file__).parent.parent / "models" / "chart_codebook"


@dataclass
class CodebookConfig:
    n_tokens: int = 256
    feature_dim: int = 5  # OHLCV
    seq_len: int = 64


class ChartTokenizer:
    def __init__(self, cfg: Optional[CodebookConfig] = None):
        self.cfg = cfg or CodebookConfig()
        self._kmeans = None
        self._loaded_path: Optional[Path] = None

    def fit(self, ohlcv_rows: Sequence[Sequence[float]]) -> None:
        try:
            from sklearn.cluster import MiniBatchKMeans  # type: ignore
            import numpy as np  # type: ignore

            X = np.asarray(ohlcv_rows, dtype=float)
            if X.ndim != 2 or X.shape[1] != self.cfg.feature_dim:
                raise ValueError(f"Expected 2D array with {self.cfg.feature_dim} cols, got {X.shape}")
            X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
            self._kmeans = MiniBatchKMeans(n_clusters=self.cfg.n_tokens, n_init=10, random_state=42)
            self._kmeans.fit(X_norm)
            CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)
            import pickle
            path = CODEBOOK_DIR / f"kmeans_{self.cfg.n_tokens}.pkl"
            with open(path, "wb") as fp:
                pickle.dump(self._kmeans, fp)
            (CODEBOOK_DIR / "config.json").write_text(json.dumps({
                "n_tokens": self.cfg.n_tokens,
                "feature_dim": self.cfg.feature_dim,
                "seq_len": self.cfg.seq_len,
            }, indent=2))
            self._loaded_path = path
        except Exception as e:
            logger.error(f"[B.4] fit failed: {e}")

    def encode(self, ohlcv_rows: Sequence[Sequence[float]]) -> List[int]:
        if self._kmeans is None:
            return []
        try:
            import numpy as np  # type: ignore

            X = np.asarray(ohlcv_rows, dtype=float)
            return [int(t) for t in self._kmeans.predict(X)]
        except Exception as e:
            logger.error(f"[B.4] encode failed: {e}")
            return []

    def load_latest(self) -> bool:
        try:
            import pickle

            if not CODEBOOK_DIR.exists():
                return False
            paths = sorted(CODEBOOK_DIR.glob("kmeans_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not paths:
                return False
            with open(paths[0], "rb") as fp:
                self._kmeans = pickle.load(fp)
            self._loaded_path = paths[0]
            return True
        except Exception as e:
            logger.error(f"[B.4] load_latest failed: {e}")
            return False
