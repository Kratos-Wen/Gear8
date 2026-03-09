"""Deploy stage runtime with behavior aligned to User_Study_AVCR.py."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import faiss
import numpy as np
import pygame
import pyttsx3
import sounddevice as sd
import torch
import torch.nn.functional as F
import whisper
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from torchvision.transforms import Compose
from ultralytics import YOLO

from .config import PipelineConfig
from .query_acquisition import equalize_frame, merge_detections, suppress_duplicate_boxes
from .retrieval_augmented_multimodal_interaction import RetrievalAugmentedMultimodalInteraction
from .speech_input_output import VoiceQueryHandler, text_to_speech


@contextmanager
def _temporary_working_directory(target_dir: Path):
    """Temporarily switch current working directory."""
    previous_dir = Path.cwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


class DeployStageRuntime:
    """Modularized deploy-stage runtime equivalent to the original implementation."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

        # Runtime state equivalent to original globals.
        self.dialogue_history = deque(maxlen=self.config.interaction.history_size)
        self.stop_audio_flag = False
        self.redo_detection = False
        self.object_infos = []

        # Device and audio samplerate (same behavior as original script).
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_samplerate = int(sd.query_devices(kind="input")["default_samplerate"])
        print(f"Using default input samplerate: {self.default_samplerate}")

        self._load_models()
        self._setup_ui_and_audio()
        self._setup_interaction_module()

    def _load_models(self) -> None:
        """Load detector, depth model, STT, embedding model, LLM, and KB index."""
        self.model = YOLO(str(self.config.paths.yolo_weights))

        depth_anything_root = self.config.paths.depth_anything_root
        depth_pkg_dir = depth_anything_root / "depth_anything"
        if not depth_pkg_dir.exists():
            raise FileNotFoundError(
                "Depth-Anything source is missing. Expected a directory containing "
                f"`depth_anything/` at: {depth_anything_root}"
            )
        depth_root_str = str(depth_anything_root.resolve())
        if depth_root_str not in sys.path:
            sys.path.insert(0, depth_root_str)

        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

        # Depth-Anything expects local torchhub paths relative to its repository root.
        with _temporary_working_directory(depth_anything_root):
            self.depth_model = DepthAnything.from_pretrained(self.config.models.depth_model_name).to(self.device).eval()

        self.speech_model = whisper.load_model(self.config.models.stt_model_name).to(self.device)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", self.config.interaction.tts_rate)

        self.embedding_model = SentenceTransformer(self.config.models.embedding_model_name)
        self.llm = Llama(
            model_path=str(self.config.paths.llm_model),
            n_ctx=self.config.llm.context_tokens,
            n_threads=self.config.llm.cpu_threads,
            n_gpu_layers=self.config.llm.gpu_layers,
            verbose=False,
        )

        with open(self.config.paths.components_json, "r", encoding="utf-8") as f:
            database = json.load(f)
        self.texts = [json.dumps(info) for info in database.values()]
        embeddings = self.embedding_model.encode(self.texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def _setup_ui_and_audio(self) -> None:
        """Initialize pygame and event window exactly as original behavior."""
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((200, 200))
        pygame.display.set_caption("Control Panel")
        pygame.mixer.init()

    def _should_stop_audio(self) -> bool:
        """Expose stop flag for speech helper."""
        return self.stop_audio_flag

    def _setup_interaction_module(self) -> None:
        """Build modular interaction wrapper and voice handler."""
        self.interaction_module = RetrievalAugmentedMultimodalInteraction(
            llm=self.llm,
            tts_engine=self.tts_engine,
            embedding_model=self.embedding_model,
            dialogue_history_ref=lambda: self.dialogue_history,
            object_infos_ref=lambda: self.object_infos,
            stop_audio_fn=self._should_stop_audio,
        )
        self.voice_handler = VoiceQueryHandler(
            speech_model=self.speech_model,
            callback=self.process_query,
            samplerate=self.default_samplerate,
            duration=self.config.interaction.query_duration_sec,
        )

    def process_query(self, query: str) -> None:
        """Process one query with original prompt and memory semantics."""
        self.interaction_module.process_query(
            query=query,
            max_tokens=256,
            temperature=0.2,
            stop_token="<end>",
        )

    def _run_query_acquisition(self):
        """Run camera capture and multi-frame detection aggregation."""
        cap = cv2.VideoCapture(self.config.query.camera_index)
        consecutive_count = 0
        frame_buffer = deque(maxlen=self.config.query.max_saved_frames)
        all_detections = []
        print("Starting camera detection...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.config.query.frame_equalize:
                frame = equalize_frame(frame)

            results = self.model(frame, augment=True)
            frame_detections = []
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result.tolist()
                if conf < self.config.query.confidence_threshold:
                    continue
                bbox_width = int(x2 - x1)
                bbox_height = int(y2 - y1)
                label = results[0].names[int(cls)]
                frame_detections.append(
                    {
                        "label": label,
                        "bbox": {"x": int(x1), "y": int(y1), "width": bbox_width, "height": bbox_height},
                        "confidence": float(conf),
                    }
                )

            if len(frame_detections) > 0:
                consecutive_count += 1
            else:
                consecutive_count = 0

            frame_buffer.append(frame)
            all_detections.append(frame_detections)
            if consecutive_count >= self.config.query.required_consecutive_frames:
                print(
                    f"Detected objects for {self.config.query.required_consecutive_frames} "
                    "consecutive frames. Exiting capture loop..."
                )
                break
        cap.release()
        return frame_buffer, all_detections

    def _merge_and_visualize(self, frame_buffer, all_detections):
        """Merge detections and save merged visualization as in original script."""
        final_results = merge_detections(
            all_detections,
            iou_threshold=self.config.query.iou_threshold,
            min_votes=self.config.query.min_votes,
        )
        final_results = suppress_duplicate_boxes(final_results)

        if len(frame_buffer) > 0:
            output_frame = frame_buffer[-1].copy()
        else:
            output_frame = np.zeros((512, 512, 3), dtype=np.uint8)

        for det in final_results:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x"]), int(bbox["y"])
            x2, y2 = x1 + int(bbox["width"]), y1 + int(bbox["height"])
            label = det["label"]
            conf = det["confidence"]
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(
                output_frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imwrite(str(self.config.paths.merged_output_image), output_frame)
        print(f"Merged visualization saved to {self.config.paths.merged_output_image}")
        return final_results, output_frame

    def _announce_detected_labels(self, final_results):
        """Announce detected labels using original string formatting."""
        labels = [det["label"] for det in final_results]
        if labels:
            label_text = "Detected objects: " + ", ".join(labels)
        else:
            label_text = "No objects detected."
        print(label_text)
        text_to_speech(label_text, self.tts_engine, self._should_stop_audio)

    def _build_object_infos(self, final_results, output_frame):
        """Run depth estimation and create object_infos exactly as original script."""
        if len(final_results) > 0:
            image_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image_rgb.shape[:2]
            image_input = self.transform({"image": image_rgb})["image"]
            image_input = torch.from_numpy(image_input).unsqueeze(0).to(self.device)
            with torch.no_grad():
                depth_map = self.depth_model(image_input)
            depth_map = F.interpolate(depth_map[None], (h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

            self.object_infos = []
            for det in final_results:
                bbox = det["bbox"]
                label = det["label"]
                conf = det["confidence"]
                x1, y1 = int(bbox["x"]), int(bbox["y"])
                center_x = int(x1 + bbox["width"] / 2)
                center_y = int(y1 + bbox["height"] / 2)
                center_depth = float(depth_map[center_y, center_x])
                query_embedding = self.embedding_model.encode([label])
                _, i = self.index.search(query_embedding, k=2)
                retrieved_parts = [self.texts[idx] for idx in i[0]]
                self.object_infos.append(
                    {
                        "label": label,
                        "bbox": bbox,
                        "confidence": conf,
                        "center_depth": center_depth,
                        "info": retrieved_parts,
                    }
                )
            self.object_infos.sort(key=lambda x: x["center_depth"])
        else:
            self.object_infos = []

    def _interaction_phase(self):
        """Keyboard event loop equivalent to original behavior."""
        print("Awaiting user query:")
        print("    Press 'v' to start voice input query")
        print("    Press 'r' to restart detection")
        print("    Press 'q' to exit the program")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("Exiting program.")
                        exit(0)
                    if event.key == pygame.K_r:
                        print("Restarting detection process...")
                        self.redo_detection = True
                    if event.key == pygame.K_v:
                        print("Starting voice input query...")
                        self.voice_handler.recognize_and_handle()
            if self.redo_detection:
                break
            time.sleep(self.config.interaction.poll_interval_sec)

    def run(self) -> None:
        """Main detection-query loop matching original control flow."""
        while True:
            self.redo_detection = False
            self.stop_audio_flag = False

            frame_buffer, all_detections = self._run_query_acquisition()
            final_results, output_frame = self._merge_and_visualize(frame_buffer, all_detections)
            self._announce_detected_labels(final_results)
            self._build_object_infos(final_results, output_frame)

            self._interaction_phase()
            if self.redo_detection:
                continue
            else:
                break

        print("Program terminated.")


def main(config: PipelineConfig | None = None) -> None:
    """CLI entrypoint."""
    app = DeployStageRuntime(config=config)
    app.run()
