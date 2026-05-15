from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import supervision as sv
import torch


def _autocast_context():
    if not torch.cuda.is_available():
        return nullcontext()

    capability = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _empty_detections(height: int, width: int) -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        mask=np.empty((0, height, width), dtype=bool),
        tracker_id=np.empty((0,), dtype=np.int32),
    )


class SAM2Tracker:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._prompted = False

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")

        if detections.tracker_id is None:
            detections.tracker_id = np.arange(1, len(detections) + 1, dtype=np.int32)

        self.reset()

        with torch.inference_mode(), _autocast_context():
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
                bbox = np.asarray([xyxy], dtype=np.float32)
                self.predictor.add_new_prompt(frame_idx=0, obj_id=int(obj_id), bbox=bbox)

        self._prompted = True

    def propagate(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        height, width = frame.shape[:2]
        with torch.inference_mode(), _autocast_context():
            tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        if len(tracker_ids) == 0:
            return _empty_detections(height, width)

        masks = (mask_logits > 0.0).detach().cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]

        count = min(len(tracker_ids), len(masks))
        tracker_ids = tracker_ids[:count]
        masks = masks[:count]

        if count == 0:
            return _empty_detections(height, width)

        xyxy = sv.mask_to_xyxy(masks=masks)
        return sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)

    def reset(self) -> None:
        for method_name in ("reset_state", "reset"):
            method = getattr(self.predictor, method_name, None)
            if callable(method):
                method()
                break
        self._prompted = False
