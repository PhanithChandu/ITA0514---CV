from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.rest import router as rest_router
from app.api.websocket import router as ws_router
from app.services.yolo_service import worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle: starts YOLO thread worker on boot, stops on shutdown."""
    worker.start()
    yield
    worker.stop()


app = FastAPI(
    title="YOLO Vision Pro — Enterprise Object Detection Engine",
    description="Production-grade real-time and static object detection platform powered by YOLOv11 & FastAPI",
    version="2.0.0",
    lifespan=lifespan,
)

# Mount static assets
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(rest_router)
app.include_router(ws_router)
