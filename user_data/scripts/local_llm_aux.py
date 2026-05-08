"""Phase 30 D.5 — Local ternary-quantized LLM auxiliary client.

Wraps llama.cpp (or compatible) for local inference of a small, cheap model
used for compaction, title generation, log compression. NOT used for trade
decisions.

If llama-cpp-python is not installed, falls back to no-op (returns input
truncated). Real model: BitNet b1.58-2B at user_data/models/bitnet/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "bitnet" / "bitnet-b158-2b.q4_k_m.gguf"
MAX_TOKENS_DEFAULT = 256


class LocalLLMAux:
    def __init__(self, weights_path: Path = WEIGHTS_PATH, n_ctx: int = 4096):
        self.weights_path = Path(weights_path)
        self.n_ctx = int(n_ctx)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.weights_path.is_file():
            return None
        try:
            from llama_cpp import Llama  # type: ignore

            self._model = Llama(model_path=str(self.weights_path), n_ctx=self.n_ctx,
                                n_threads=4, verbose=False)
            return self._model
        except Exception as e:
            logger.warning(f"[D.5] llama_cpp unavailable: {e}")
            return None

    def compress(self, text: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        m = self._load()
        if m is None:
            return text[: max_tokens * 4]
        try:
            out = m(prompt=f"Compress to <={max_tokens} tokens, preserve key facts:\n{text}\n\nCompressed:",
                    max_tokens=max_tokens, stop=["\n\n"])
            return out["choices"][0]["text"].strip() if out else text[: max_tokens * 4]
        except Exception as e:
            logger.error(f"[D.5] inference failed: {e}")
            return text[: max_tokens * 4]

    def title(self, text: str) -> str:
        return self.compress(f"Generate one short title for:\n{text}", max_tokens=20)


_GLOBAL: Optional[LocalLLMAux] = None


def get_local_aux() -> LocalLLMAux:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = LocalLLMAux()
    return _GLOBAL
