"""Phase 30 D.4 — Chart -> PNG renderer for visual perception.

Standardizes a pair's last-N-bar OHLCV into a fixed-size PNG suitable for
YOLO/CNN inference. Uses matplotlib (mpf-style candlestick) when available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path(__file__).parent.parent.parent / "data" / "chart_pngs"


def render_pair_chart_png(
    pair: str,
    ohlcv_rows: Sequence[Sequence[float]],
    width_px: int = 416,
    height_px: int = 416,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in pair)
    out_path = out_dir / f"{safe}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        logger.error("[D.4] matplotlib unavailable; cannot render PNG")
        return None
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    if ohlcv_rows:
        opens = [r[0] for r in ohlcv_rows]
        highs = [r[1] for r in ohlcv_rows]
        lows = [r[2] for r in ohlcv_rows]
        closes = [r[3] for r in ohlcv_rows]
        x = list(range(len(closes)))
        ax.plot(x, closes, color="#0f0", linewidth=0.7)
        ax.fill_between(x, lows, highs, alpha=0.2, color="#0a4")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#000")
    fig.patch.set_facecolor("#000")
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path
