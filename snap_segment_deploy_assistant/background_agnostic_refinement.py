"""Background-Agnostic Refinement (BAR) utilities for detector training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class BARConfig:
    """Stage-2 pseudo-label filtering settings."""

    confidence_threshold: float = 0.5
    white_canvas_value: int = 255


def crop_predictions_to_white_canvas(
    image_path: Path,
    detections: list[tuple[int, int, int, int, float]],
    output_dir: Path,
    config: BARConfig | None = None,
) -> list[Path]:
    """
    Create stage-2 BAR crops by removing background context.

    Args:
        image_path: Source training image.
        detections: List of (x1, y1, x2, y2, confidence).
        output_dir: Destination folder for pseudo-labeled crops.
        config: Optional BARConfig.
    """
    cfg = config or BARConfig()
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    image_stem = image_path.stem

    for idx, (x1, y1, x2, y2, confidence) in enumerate(detections):
        if confidence < cfg.confidence_threshold:
            continue
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        canvas = cv2.cvtColor(
            cv2.threshold(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                0,
                cfg.white_canvas_value,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1],
            cv2.COLOR_GRAY2BGR,
        )
        canvas[:] = cfg.white_canvas_value
        canvas[0 : crop.shape[0], 0 : crop.shape[1]] = crop

        out_path = output_dir / f"{image_stem}_bar_{idx:03d}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)

    return saved_paths

