import asyncio
import json
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.yolo_service import worker

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/detect")
async def detect_ws(ws: WebSocket):
    """High-throughput, real-time WebSocket endpoint for streaming video detection."""
    await ws.accept()
    log.info("WebSocket client session opened.")

    running = True

    async def _receive():
        nonlocal running
        try:
            while running:
                msg = await ws.receive()

                if msg["type"] == "websocket.disconnect":
                    break

                # Control message (JSON)
                if msg.get("text"):
                    try:
                        ctrl = json.loads(msg["text"])
                        if "conf" in ctrl:
                            worker.conf_threshold = float(ctrl["conf"])
                        if "iou" in ctrl:
                            worker.iou_threshold = float(ctrl["iou"])
                        if "enabled" in ctrl:
                            worker.detection_enabled = bool(ctrl["enabled"])
                    except Exception as e:
                        log.warning(f"Error parsing WebSocket control payload: {e}")
                    continue

                # Binary frame (JPEG bytes)
                raw = msg.get("bytes")
                if not raw:
                    continue

                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                worker.submit_frame(frame, time.time())

        except Exception as e:
            log.info(f"WebSocket receiver session closed: {e}")
        finally:
            running = False

    async def _send():
        sleep_ms = 0.001
        while running:
            result = worker.get_result()
            if result:
                try:
                    await ws.send_text(json.dumps(result))
                    sleep_ms = 0.001
                except Exception:
                    break
            else:
                await asyncio.sleep(sleep_ms)
                sleep_ms = min(sleep_ms * 1.5, 0.008)

    try:
        await asyncio.gather(_receive(), _send())
    except WebSocketDisconnect:
        log.info("Client disconnected from WebSocket.")
    except Exception as e:
        log.info(f"WebSocket session terminated: {e}")
    finally:
        running = False
        log.info("WebSocket session closed successfully.")
