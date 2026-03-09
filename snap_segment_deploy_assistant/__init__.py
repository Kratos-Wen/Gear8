"""Snap-Segment-Deploy assistant package for arXiv:2507.21072v1."""

from .config import PipelineConfig
from .deploy_stage_runtime import DeployStageRuntime, main

__all__ = ["PipelineConfig", "DeployStageRuntime", "main"]

