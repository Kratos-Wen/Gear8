"""Configuration for the Snap-Segment-Deploy assistant pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates: list[Path]) -> Path:
    """Return the first existing path, otherwise fall back to the first candidate."""
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


FASTSAM_ROOT = _first_existing(
    [
        REPO_ROOT / "FastSAM_Cutie" / "FastSAM",
        REPO_ROOT.parent / "FastSAM_Cutie" / "FastSAM",
        REPO_ROOT.parents[1] / "FastSAM_Cutie" / "FastSAM",
    ]
)

# Fixed single location for the Depth-Anything codebase.
DEPTH_ANYTHING_ROOT = REPO_ROOT / "third_party" / "Depth-Anything"


@dataclass(frozen=True)
class ModuleNameConfig:
    """Canonical module names from 2507.21072v1."""

    stage_1: str = "Snap"
    stage_2: str = "Segment"
    stage_3: str = "Deploy"
    training_module: str = "Background-Agnostic Refinement (BAR)"
    interaction_module: str = "Retrieval-Augmented Multimodal Interaction"
    kb_module: str = "Knowledge Base Construction"
    query_module: str = "Query Acquisition"
    response_module: str = "Semantic Retrieval and Knowledge-Augmented Response Generation"


@dataclass(frozen=True)
class PathConfig:
    """File paths for checkpoints and metadata."""

    yolo_weights: Path = FASTSAM_ROOT / "runs_2stage_small" / "YOLO11s2" / "weights" / "best.pt"
    llm_model: Path = FASTSAM_ROOT / "Phi-3-mini-4k-instruct-Q6_K.gguf"
    components_json: Path = FASTSAM_ROOT / "components.json"
    depth_anything_root: Path = DEPTH_ANYTHING_ROOT
    merged_output_image: Path = Path("merged_output.jpg")


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers used by the original implementation."""

    depth_model_name: str = "LiheYoung/depth_anything_vitl14"
    stt_model_name: str = "base"
    embedding_model_name: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class QueryAcquisitionConfig:
    """Detection and multi-frame fusion settings."""

    camera_index: int = 0
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.5
    min_votes: int = 2
    required_consecutive_frames: int = 5
    max_saved_frames: int = 10
    frame_equalize: bool = False


@dataclass(frozen=True)
class InteractionConfig:
    """Speech and runtime interaction settings."""

    query_duration_sec: int = 8
    fallback_samplerate: int = 16000
    tts_rate: int = 180
    history_size: int = 20
    history_top_k: int = 2
    kb_top_k: int = 2
    poll_interval_sec: float = 0.1


@dataclass(frozen=True)
class LLMConfig:
    """Local LLM runtime options."""

    context_tokens: int = 4096
    cpu_threads: int = 4
    gpu_layers: int = 100
    max_tokens: int = 256
    temperature: float = 0.2
    stop_token: str = "<end>"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level runtime configuration."""

    modules: ModuleNameConfig = field(default_factory=ModuleNameConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    query: QueryAcquisitionConfig = field(default_factory=QueryAcquisitionConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    run_tag: str = "ssd_baseline"
