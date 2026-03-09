"""Segment stage: simple synthetic composition helper."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np


def compose_on_background(
    background_path: Path,
    object_crops: list[Path],
    output_path: Path,
    min_objects: int = 3,
    max_objects: int = 5,
) -> bool:
    """Create one synthetic image by compositing object crops onto a background."""
    background = cv2.imread(str(background_path))
    if background is None or not object_crops:
        return False

    canvas = background.copy()
    h, w = canvas.shape[:2]
    num_objects = random.randint(min_objects, max_objects)

    chosen = random.sample(object_crops, k=min(num_objects, len(object_crops)))
    for crop_path in chosen:
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        scale = random.uniform(0.5, 1.0)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ch, cw = crop.shape[:2]
        if ch >= h or cw >= w:
            continue

        x = random.randint(0, w - cw - 1)
        y = random.randint(0, h - ch - 1)

        mask = np.any(crop < 250, axis=2)
        roi = canvas[y : y + ch, x : x + cw]
        roi[mask] = crop[mask]
        canvas[y : y + ch, x : x + cw] = roi

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return True

