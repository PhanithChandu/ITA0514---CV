from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.services.yolo_service import worker

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Renders the main enterprise dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/health")
async def health_check():
    """System health check and inference performance diagnostic endpoint."""
    avg_fps = (sum(worker.fps_history) / len(worker.fps_history)) if worker.fps_history else 0.0
    return {
        "status": "healthy",
        "model": worker.model_path,
        "worker_running": worker._running,
        "avg_fps": round(avg_fps, 1),
        "conf_threshold": worker.conf_threshold,
        "iou_threshold": worker.iou_threshold,
        "detection_enabled": worker.detection_enabled,
    }


@router.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(None, ge=0.01, le=1.0, description="Confidence threshold"),
    iou: Optional[float] = Query(None, ge=0.01, le=1.0, description="IoU threshold"),
):
    """REST endpoint for single static image object detection."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        contents = await file.read()
        res = worker.detect_image_bytes(contents, conf=conf, iou=iou)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
