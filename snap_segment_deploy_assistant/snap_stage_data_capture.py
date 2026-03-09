"""Snap stage: multi-view data capture helpers."""

from __future__ import annotations

from pathlib import Path

import cv2


def sample_frames_from_video(video_path: Path, output_dir: Path, every_n_frames: int = 15) -> list[Path]:
    """Sample frames from one part video for the dataset construction stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frame_idx = 0
    saved_paths: list[Path] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % every_n_frames == 0:
                out_path = output_dir / f"{video_path.stem}_{frame_idx:06d}.png"
                cv2.imwrite(str(out_path), frame)
                saved_paths.append(out_path)
            frame_idx += 1
    finally:
        capture.release()

    return saved_paths

