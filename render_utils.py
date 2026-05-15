from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderOptions:
    selected_team: int = 0
    spotlight_my_team: bool = True
    blur_opposition: bool = False
    team_colour_overlay: bool = True
    show_player_ids: bool = False


def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16)


def _normalise_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _team_for_track(teams_by_track: np.ndarray, tracker_id: int) -> int | None:
    index = int(tracker_id) - 1
    if index < 0 or index >= len(teams_by_track):
        return None
    return int(teams_by_track[index])


def _draw_id(frame: np.ndarray, mask: np.ndarray, tracker_id: int) -> None:
    if not mask.any():
        return

    ys, xs = np.where(mask)
    x = int(xs.mean())
    y = max(int(ys.min()) - 8, 18)
    label = f"#{int(tracker_id)}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top_left = (max(x - text_width // 2 - 4, 0), max(y - text_height - 5, 0))
    bottom_right = (
        min(top_left[0] + text_width + 8, frame.shape[1] - 1),
        min(top_left[1] + text_height + baseline + 8, frame.shape[0] - 1),
    )
    text_origin = (top_left[0] + 4, bottom_right[1] - baseline - 3)

    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(frame, label, text_origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - converted into a clearer runtime error.
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or imageio-ffmpeg.") from exc


def compress_video(raw_path: Path, final_path: Path) -> Path:
    command = [
        _ffmpeg_executable(),
        "-y",
        "-i",
        str(raw_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(final_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to encode the output video: {result.stderr.strip()}")
    return final_path


def cleanup_old_outputs(output_dir: Path | str, max_age_seconds: int = 6 * 60 * 60) -> None:
    directory = Path(output_dir)
    if not directory.exists():
        return

    cutoff = time.time() - max_age_seconds
    for path in directory.glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def render_tactical_view(state: Any, options: RenderOptions, output_dir: Path | str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames = state.frames
    fps = float(state.fps or 30.0)
    width = int(state.width)
    height = int(state.height)
    teams_by_track = np.asarray(state.teams_by_track)
    team_colours = tuple(state.team_colours)
    colour_lookup = {
        0: _hex_to_bgr(team_colours[0]),
        1: _hex_to_bgr(team_colours[1]),
    }

    render_id = uuid.uuid4().hex
    raw_path = output_path / f"{Path(state.source_name).stem}-{render_id}-raw.mp4"
    final_path = output_path / f"{Path(state.source_name).stem}-{render_id}.mp4"

    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the output video.")

    try:
        for index, frame_rgb in enumerate(frames):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result = frame_bgr.copy()

            masks = state.masks_per_frame[index] if index < len(state.masks_per_frame) else []
            tracker_ids = (
                state.tracker_ids_per_frame[index]
                if index < len(state.tracker_ids_per_frame)
                else []
            )

            selected_union = np.zeros((height, width), dtype=bool)
            opposition_union = np.zeros((height, width), dtype=bool)
            normalised_masks: list[tuple[np.ndarray, int, int]] = []

            for mask, tracker_id in zip(masks, tracker_ids):
                team_id = _team_for_track(teams_by_track, int(tracker_id))
                if team_id is None:
                    continue

                mask = _normalise_mask(mask, height, width)
                normalised_masks.append((mask, int(tracker_id), team_id))
                if team_id == options.selected_team:
                    selected_union |= mask
                else:
                    opposition_union |= mask

            if options.spotlight_my_team and selected_union.any():
                dimmed = (result * 0.3).astype(np.uint8)
                result = np.where(selected_union[..., None], result, dimmed)

            if options.blur_opposition and opposition_union.any():
                blurred = cv2.GaussianBlur(result, (31, 31), 0)
                result = np.where(opposition_union[..., None], blurred, result)

            if options.team_colour_overlay and normalised_masks:
                overlay = result.copy()
                for mask, _, team_id in normalised_masks:
                    overlay[mask] = colour_lookup.get(team_id, (255, 255, 255))
                result = cv2.addWeighted(result, 0.55, overlay, 0.45, 0)

            if options.show_player_ids:
                for mask, tracker_id, _ in normalised_masks:
                    _draw_id(result, mask, tracker_id)

            writer.write(result)
    finally:
        writer.release()

    try:
        compress_video(raw_path, final_path)
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass

    return str(final_path)
