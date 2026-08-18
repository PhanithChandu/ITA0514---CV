import base64
import logging
import queue
import threading
import time
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import (
    MODEL_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_INFER_SIZE,
    JPEG_QUALITY,
    LATENCY_THRESHOLD_MS,
)

log = logging.getLogger(__name__)

# Configure OpenCV thread count for optimal multi-core inference & encoding
cv2.setNumThreads(0)

# Curated palette for class visualization
COLOR_PALETTE = [
    (249, 115, 22),   # Vibrant Orange
    (16, 185, 129),   # Emerald Green
    (59, 130, 246),   # Electric Blue
    (236, 72, 153),   # Pink
    (168, 85, 247),   # Purple
    (234, 179, 8),    # Amber Yellow
    (6, 182, 212),    # Cyan
    (244, 63, 94),    # Rose Red
]


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """Generates consistent RGB color based on label string hash."""
    val = sum(ord(c) for c in label)
    return COLOR_PALETTE[val % len(COLOR_PALETTE)]


class YOLOWorker:
    """
    CPU-Optimized YOLO Engine
    -------------------------
    - Pre-resizes frames to target infer size (320x320) for 3-4x speedup on CPU.
    - Accurately projects bounding box coordinates back to original frame dimensions.
    - Non-blocking ring-buffer queue system prevents frame lag.
    - Server-side drawing on original frame preserves native aspect ratio and image clarity.
    """

    INPUT_QUEUE_DEPTH = 1
    OUTPUT_QUEUE_DEPTH = 2

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self._in_q: queue.Queue = queue.Queue(maxsize=self.INPUT_QUEUE_DEPTH)
        self._out_q: queue.Queue = queue.Queue(maxsize=self.OUTPUT_QUEUE_DEPTH)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self.fps_history: deque = deque(maxlen=60)
        self.conf_threshold: float = DEFAULT_CONF_THRESHOLD
        self.iou_threshold: float = DEFAULT_IOU_THRESHOLD
        self.detection_enabled: bool = True
        self.infer_size: int = DEFAULT_INFER_SIZE

        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    def start(self):
        log.info(f"Loading YOLO model weights from: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Warmup execution to compile PyTorch/JIT kernels
        dummy = np.zeros((self.infer_size, self.infer_size, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy, verbose=False, imgsz=self.infer_size)
        log.info("YOLO model initialized and warmed up successfully.")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="yolo-worker")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def submit_frame(self, frame: np.ndarray, ts: float) -> bool:
        """Submits frame to processing queue, dropping stale frames if busy."""
        try:
            self._in_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._in_q.put_nowait((frame, ts))
            return True
        except queue.Full:
            return False

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Retrieves newest processed inference output."""
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def _loop(self):
        while self._running:
            try:
                frame, ts = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            lag_ms = (time.time() - ts) * 1000
            if lag_ms > LATENCY_THRESHOLD_MS:
                log.debug(f"Skipping stale queued frame (lag: {lag_ms:.0f}ms)")
                continue

            t0 = time.perf_counter()
            orig_h, orig_w = frame.shape[:2]

            if self.detection_enabled and self.model is not None:
                # Pre-resize frame to inference grid
                small = cv2.resize(frame, (self.infer_size, self.infer_size), interpolation=cv2.INTER_LINEAR)
                results = self.model(
                    small,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    imgsz=self.infer_size,
                )[0]

                dets = self._parse_and_scale(results, orig_w, orig_h, self.infer_size, self.infer_size)
                annotated = self.annotate_frame(frame.copy(), dets)
            else:
                annotated = frame
                dets = []

            elapsed = time.perf_counter() - t0
            self.fps_history.append(1.0 / max(elapsed, 1e-6))

            _, buf = cv2.imencode(".jpg", annotated, self._encode_params)
            b64_img = base64.b64encode(buf).decode()

            payload = {
                "image": b64_img,
                "detections": dets,
                "fps": round(sum(self.fps_history) / len(self.fps_history), 1),
                "inference_ms": round(elapsed * 1000, 1),
                "frame_width": orig_w,
                "frame_height": orig_h,
            }

            try:
                self._out_q.put_nowait(payload)
            except queue.Full:
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._out_q.put_nowait(payload)
                except queue.Full:
                    pass

    @staticmethod
    def _parse_and_scale(results, orig_w: int, orig_h: int, infer_w: int, infer_h: int) -> List[Dict[str, Any]]:
        """Parses model boxes and rescales coordinates from inference grid to original image bounds."""
        out = []
        if results.boxes is None:
            return out

        scale_x = orig_w / float(infer_w)
        scale_y = orig_h / float(infer_h)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            x1 = int(round(xyxy[0] * scale_x))
            y1 = int(round(xyxy[1] * scale_y))
            x2 = int(round(xyxy[2] * scale_x))
            y2 = int(round(xyxy[3] * scale_y))

            # Clamp coordinates to original bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            out.append({
                "label": results.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            })
        return out

    @staticmethod
    def annotate_frame(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws crisp bounding boxes, filled headers, and confidence tags on the target frame."""
        h, w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            color = get_color_for_label(label)

            # Draw outer rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Header label badge
            text = f"{label} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = max(text_h + 8, y1)
            cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 12, badge_y2), color, -1)

            # Text string
            cv2.putText(
                frame,
                text,
                (x1 + 6, badge_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        return frame

    def detect_image_bytes(
        self, image_bytes: bytes, conf: Optional[float] = None, iou: Optional[float] = None
    ) -> Dict[str, Any]:
        """Synchronous detection helper for REST API image uploads."""
        if self.model is None:
            log.info(f"Lazy loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image buffer.")

        orig_h, orig_w = frame.shape[:2]
        conf_val = conf if conf is not None else self.conf_threshold
        iou_val = iou if iou is not None else self.iou_threshold

        t0 = time.perf_counter()
        small = cv2.resize(frame, (self.infer_size, self.infer_size), interpolation=cv2.INTER_LINEAR)
        results = self.model(
            small,
            conf=conf_val,
            iou=iou_val,
            verbose=False,
            imgsz=self.infer_size,
        )[0]

        dets = self._parse_and_scale(results, orig_w, orig_h, self.infer_size, self.infer_size)
        annotated = self.annotate_frame(frame.copy(), dets)
        elapsed = time.perf_counter() - t0

        _, buf = cv2.imencode(".jpg", annotated, self._encode_params)
        b64_img = base64.b64encode(buf).decode()

        # Class counts summary
        summary = {}
        for d in dets:
            lbl = d["label"]
            summary[lbl] = summary.get(lbl, 0) + 1

        return {
            "image": b64_img,
            "detections": dets,
            "summary": summary,
            "total_objects": len(dets),
            "inference_ms": round(elapsed * 1000, 1),
            "width": orig_w,
            "height": orig_h,
        }


# Global worker instance
worker = YOLOWorker()
