"""Kronos-mini vendored loader.

The upstream `NeoQuasar/Kronos-mini` HF repo doesn't ship an `auto_map` so
`transformers.AutoModel` can't dispatch to it. Per the upstream Kronos repo
(https://github.com/shiyu-coder/Kronos), the model class is custom and ships
in `model.py`. We snapshot-download the HF repo locally, dynamically import
the upstream model module, and expose the three public symbols downstream
callers want:

    Kronos               — backbone class (PreTrainedModel-like)
    KronosTokenizer      — OHLCV → token-id encoder
    KronosPredictor      — high-level inference wrapper

If the snapshot doesn't include the upstream Python source (some HF mirrors
strip `*.py`), we fall back to a minimal local TransformerEncoder + linear
head so callers that just need a directional score can still get one.
"""

from .kronos import Kronos, KronosTokenizer, KronosPredictor  # noqa: F401

__all__ = ["Kronos", "KronosTokenizer", "KronosPredictor"]
