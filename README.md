# Snap-Segment-Deploy Code Structure

This folder reorganizes the implementation around the terminology of **arXiv:2507.21072v1**:

- **Snap**: multi-view part data capture
- **Segment**: mask-driven synthetic composition
- **Deploy**: on-device assistant runtime

The deploy runtime follows the original script behavior: YOLO-based detection, multi-frame fusion, depth-aware object ranking, FAISS retrieval, local LLM response generation, Whisper speech input, and pyttsx3 audio output.

## Module Map (Paper-Aligned Names)

- `background_agnostic_refinement.py`: **Background-Agnostic Refinement (BAR)**
- `knowledge_base_construction.py`: **Knowledge Base Construction**
- `query_acquisition.py`: **Query Acquisition**
- `semantic_retrieval_and_response_generation.py`: **Semantic Retrieval and Knowledge-Augmented Response Generation**
- `retrieval_augmented_multimodal_interaction.py`: **Retrieval-Augmented Multimodal Interaction**
- `deploy_stage_runtime.py`: **Deploy** runtime integration
- `snap_stage_data_capture.py`: **Snap** helpers
- `segment_stage_synthetic_composition.py`: **Segment** helpers

## Folder Structure

```text
snap_segment_deploy/
├── README.md
├── run_deploy_stage.py
└── snap_segment_deploy_assistant/
    ├── __init__.py
    ├── config.py
    ├── types.py
    ├── background_agnostic_refinement.py
    ├── snap_stage_data_capture.py
    ├── segment_stage_synthetic_composition.py
    ├── knowledge_base_construction.py
    ├── query_acquisition.py
    ├── semantic_retrieval_and_response_generation.py
    ├── retrieval_augmented_multimodal_interaction.py
    ├── speech_input_output.py
    └── deploy_stage_runtime.py
```

## Default Runtime Models and Paths

Defaults in `config.py` are aligned to the original local setup:

- YOLO weights: `FastSAM_Cutie/FastSAM/runs_2stage_small/YOLO11s2/weights/best.pt`
- Depth model: `LiheYoung/depth_anything_vitl14`
- Embedding model: `all-MiniLM-L6-v2`
- STT model: `whisper` `"base"`
- LLM: `Phi-3-mini-4k-instruct-Q6_K.gguf`

Update paths in `snap_segment_deploy_assistant/config.py` if needed.

## Run Deploy Stage

```bash
cd ACVR/snap_segment_deploy
python run_deploy_stage.py
```

Runtime keys:

- `v`: start voice query
- `r`: restart query acquisition
- `q`: quit

## Notes

- This structure is intentionally single-agent and does not include role-routing logic.
- `background_agnostic_refinement.py` provides BAR data-preparation utilities for stage-2 training.
- `snap_stage_data_capture.py` and `segment_stage_synthetic_composition.py` are lightweight helpers for dataset pipeline scripting.
