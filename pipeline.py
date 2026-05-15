from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import supervision as sv
import torch
from huggingface_hub import hf_hub_download

from sam2_tracker import SAM2Tracker


APP_ROOT = Path(__file__).resolve().parent

RFDETR_CHECKPOINT = "checkpoints/checkpoint_best_regular.pth"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1_hiera_t.yaml"

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

DEFAULT_CHECKPOINT_REPO_ID = "zohaib/coach-lens-checkpoints"
PLAYER_DETECTION_THRESHOLD = 0.4
PLAYER_CLASS_IDS = (0,)
TARGET_WIDTH = 720
MAX_VIDEO_SECONDS = 10.0
MAX_UPLOAD_SIZE_MB = 100
TEAM_CLUSTER_STRIDE = 15
MAX_TEAM_CROPS = 400
MIN_TEAM_CROPS = 2
TEAM_COLOURS = ("#006BB6", "#007A33")


class PipelineError(RuntimeError):
    pass


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float
    resized_width: int
    resized_height: int


@dataclass
class AnalysisState:
    frames: list[np.ndarray]
    fps: float
    width: int
    height: int
    masks_per_frame: list[np.ndarray]
    tracker_ids_per_frame: list[np.ndarray]
    teams_by_track: np.ndarray
    team_colours: tuple[str, str]
    source_name: str


def _app_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else APP_ROOT / candidate


def _env_path(name: str, default: str) -> Path:
    return _app_path(os.getenv(name, default))


def _parse_player_class_ids() -> tuple[int, ...]:
    raw = os.getenv("PLAYER_CLASS_IDS")
    if not raw:
        return PLAYER_CLASS_IDS

    try:
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise PipelineError("PLAYER_CLASS_IDS must be a comma-separated list of integers.") from exc


def _ensure_runtime_dirs() -> None:
    for directory in (UPLOAD_DIR, OUTPUT_DIR, "checkpoints", "configs"):
        _app_path(directory).mkdir(parents=True, exist_ok=True)


def _download_from_model_repo(relative_path: str, target_path: Path) -> Path:
    repo_id = os.getenv("CHECKPOINT_REPO_ID", DEFAULT_CHECKPOINT_REPO_ID)
    hf_token = os.getenv("HF_TOKEN")
    errors: list[str] = []

    candidates = [relative_path.replace("\\", "/"), Path(relative_path).name]
    for filename in dict.fromkeys(candidates):
        local_dir = APP_ROOT if "/" in filename else target_path.parent
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                token=hf_token,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != target_path.resolve() and downloaded_path.name == target_path.name:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(downloaded_path.read_bytes())
            return target_path if target_path.exists() else downloaded_path
        except Exception as exc:  # noqa: BLE001 - surfaced as one deployment error below.
            errors.append(f"{filename}: {exc}")

    raise PipelineError(
        "Missing model file "
        f"{relative_path!r}. Add it locally or upload it to the Hugging Face model repo "
        f"{repo_id!r}. Download attempts failed: {' | '.join(errors)}"
    )


def _ensure_model_file(relative_path: str, env_name: str) -> Path:
    path = _env_path(env_name, relative_path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    return _download_from_model_repo(relative_path, path)


def _validate_video(video_path: str) -> VideoMetadata:
    path = Path(video_path)
    if not path.exists():
        raise PipelineError("Uploaded video file was not found.")

    if path.suffix.lower() != ".mp4":
        raise PipelineError("Please upload an MP4 file.")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise PipelineError(f"Please upload a file under {MAX_UPLOAD_SIZE_MB} MB.")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PipelineError("OpenCV could not read this video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0:
        raise PipelineError("This video has invalid dimensions.")

    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    if duration > MAX_VIDEO_SECONDS:
        raise PipelineError(f"Please upload a clip under {int(MAX_VIDEO_SECONDS)} seconds.")

    scale = min(1.0, TARGET_WIDTH / float(width))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    return VideoMetadata(
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration=duration,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def _read_video_frames(video_path: str, metadata: VideoMetadata) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    max_frames = int(np.ceil(metadata.fps * MAX_VIDEO_SECONDS))

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if metadata.resized_width != metadata.width:
            frame_bgr = cv2.resize(
                frame_bgr,
                (metadata.resized_width, metadata.resized_height),
                interpolation=cv2.INTER_AREA,
            )

        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if len(frames) > max_frames:
            cap.release()
            raise PipelineError(f"Please upload a clip under {int(MAX_VIDEO_SECONDS)} seconds.")

    cap.release()

    if not frames:
        raise PipelineError("This video does not contain readable frames.")
    return frames


def _filter_player_detections(detections: sv.Detections, player_class_ids: Iterable[int]) -> sv.Detections:
    if detections.class_id is None:
        return detections

    keep = np.isin(detections.class_id, np.asarray(tuple(player_class_ids), dtype=int))
    return detections[keep]


def _crop_players(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    if len(detections) == 0:
        return []

    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
    crops: list[np.ndarray] = []
    for box in boxes:
        crop = sv.crop_image(frame, box)
        if crop.size:
            crops.append(crop)
    return crops


class CoachLensPipeline:
    def __init__(self) -> None:
        _ensure_runtime_dirs()

        self.player_class_ids = _parse_player_class_ids()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rfdetr_checkpoint = _ensure_model_file(RFDETR_CHECKPOINT, "RFDETR_CHECKPOINT")
        self.sam2_checkpoint = _ensure_model_file(SAM2_CHECKPOINT, "SAM2_CHECKPOINT")
        self.sam2_config = _ensure_model_file(SAM2_CONFIG, "SAM2_CONFIG")

        self.detector = self._load_rfdetr()
        self.sam2_predictor = self._load_sam2_predictor()
        self.team_classifier = None

    def _load_rfdetr(self):
        from rfdetr import RFDETRBase, RFDETRLarge

        model_size = os.getenv("RFDETR_MODEL_SIZE", "base").lower()
        model_classes = {"base": RFDETRBase, "large": RFDETRLarge}
        if model_size not in model_classes:
            raise PipelineError("RFDETR_MODEL_SIZE must be 'base' or 'large'.")

        return model_classes[model_size](pretrain_weights=str(self.rfdetr_checkpoint))

    def _load_sam2_predictor(self):
        sam2_repo_path = os.getenv("SAM2_REPO_PATH")
        if sam2_repo_path and sam2_repo_path not in sys.path:
            sys.path.insert(0, sam2_repo_path)

        try:
            from sam2.build_sam import build_sam2_camera_predictor
        except ImportError as exc:
            raise PipelineError(
                "SAM2 is not importable. Install the segment-anything-2-real-time fork "
                "from requirements.txt or set SAM2_REPO_PATH to a local clone."
            ) from exc

        config_arg = os.path.relpath(self.sam2_config, APP_ROOT)
        checkpoint_arg = str(self.sam2_checkpoint)
        cwd = os.getcwd()
        try:
            os.chdir(APP_ROOT)
            return build_sam2_camera_predictor(config_arg, checkpoint_arg)
        finally:
            os.chdir(cwd)

    def _get_team_classifier(self):
        if self.team_classifier is not None:
            return self.team_classifier

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise PipelineError(
                "HF_TOKEN is not set. Add it as a Hugging Face Space secret before running team clustering."
            )

        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

        from sports import TeamClassifier

        self.team_classifier = TeamClassifier(device=self.device)
        return self.team_classifier

    def _detect_players(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detector.predict(frame, threshold=PLAYER_DETECTION_THRESHOLD)
        return _filter_player_detections(detections, self.player_class_ids)

    def _collect_team_crops(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        crops: list[np.ndarray] = []
        for index in range(0, len(frames), TEAM_CLUSTER_STRIDE):
            detections = self._detect_players(frames[index])
            crops.extend(_crop_players(frames[index], detections))
            if len(crops) >= MAX_TEAM_CROPS:
                return crops[:MAX_TEAM_CROPS]
        return crops

    def analyse(self, video_path: str) -> AnalysisState:
        metadata = _validate_video(video_path)
        frames = _read_video_frames(video_path, metadata)

        seed_detections = self._detect_players(frames[0])
        if len(seed_detections) == 0:
            raise PipelineError("No players detected in the first frame. Try a different short clip.")

        seed_detections.tracker_id = np.arange(1, len(seed_detections) + 1, dtype=np.int32)

        crops = self._collect_team_crops(frames)
        if len(crops) < MIN_TEAM_CROPS:
            raise PipelineError("Not enough player crops were found to cluster teams.")

        team_classifier = self._get_team_classifier()
        team_classifier.fit(crops)

        first_frame_crops = _crop_players(frames[0], seed_detections)
        if not first_frame_crops:
            raise PipelineError("Could not crop first-frame players for team assignment.")
        teams_by_track = np.asarray(team_classifier.predict(first_frame_crops), dtype=np.int32)

        tracker = SAM2Tracker(self.sam2_predictor)
        tracker.prompt_first_frame(frames[0], seed_detections)

        masks_per_frame: list[np.ndarray] = []
        tracker_ids_per_frame: list[np.ndarray] = []
        try:
            for frame in frames:
                tracked = tracker.propagate(frame)
                masks_per_frame.append(tracked.mask.astype(bool))
                tracker_ids_per_frame.append(tracked.tracker_id.astype(np.int32))
        finally:
            tracker.reset()

        return AnalysisState(
            frames=frames,
            fps=metadata.fps,
            width=metadata.resized_width,
            height=metadata.resized_height,
            masks_per_frame=masks_per_frame,
            tracker_ids_per_frame=tracker_ids_per_frame,
            teams_by_track=teams_by_track,
            team_colours=TEAM_COLOURS,
            source_name=Path(video_path).name,
        )


_PIPELINE: CoachLensPipeline | None = None


def get_pipeline() -> CoachLensPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = CoachLensPipeline()
    return _PIPELINE


def analyse_clip(video_path: str) -> AnalysisState:
    return get_pipeline().analyse(video_path)
