"""Phase 30 D.4 — YOLO inference runner skeleton.

Real model load expects ultralytics or onnxruntime-installed weights at
user_data/models/yolo_chart/best.onnx (out-of-session training).

Without a model, returns empty pattern list (graceful no-op).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent.parent.parent / "models" / "yolo_chart" / "best.onnx"


@dataclass
class PatternDetection:
    label: str
    confidence: float
    bbox: tuple  # (x, y, w, h) normalised


def detect_patterns(png_path: Path) -> List[PatternDetection]:
    if not WEIGHTS_PATH.is_file():
        logger.debug("[D.4] yolo weights absent; skipping inference")
        return []
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(WEIGHTS_PATH))
        results = model.predict(str(png_path), verbose=False, conf=0.4)
        out: List[PatternDetection] = []
        for r in results:
            for box, conf, cls in zip(r.boxes.xywhn, r.boxes.conf, r.boxes.cls):
                out.append(PatternDetection(
                    label=str(model.names.get(int(cls), str(int(cls)))),
                    confidence=float(conf),
                    bbox=tuple(box.tolist()),
                ))
        return out
    except Exception as e:
        logger.warning(f"[D.4] yolo runtime unavailable: {e}")
        return []
