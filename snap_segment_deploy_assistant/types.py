"""Typed data objects shared across deploy-stage modules."""

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Axis-aligned box represented by x, y, width, and height."""

    x: int
    y: int
    width: int
    height: int

    def center(self) -> tuple[int, int]:
        """Return pixel center coordinates."""
        return int(self.x + self.width / 2), int(self.y + self.height / 2)


@dataclass
class Detection:
    """One detector prediction with vote count metadata."""

    label: str
    bbox: BoundingBox
    confidence: float
    votes: int = 1


@dataclass
class RetrievedObjectContext:
    """Detection enriched with depth and retrieved database entries."""

    label: str
    bbox: BoundingBox
    confidence: float
    center_depth: float
    info: list[str]

    def to_prompt_dict(self) -> dict:
        """Convert to JSON-safe dictionary for prompt construction."""
        return {
            "label": self.label,
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "confidence": self.confidence,
            "center_depth": self.center_depth,
            "info": self.info,
        }

