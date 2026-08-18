import os

# Check for custom trained model in models/ directory, fallback to base weights
CUSTOM_MODEL_PATH = os.path.join("models", "best_cnn_model.pt")
DEFAULT_MODEL = CUSTOM_MODEL_PATH if os.path.exists(CUSTOM_MODEL_PATH) else "yolo11n.pt"

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", DEFAULT_MODEL)
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_INFER_SIZE = 640  # Increased from 320 to 640 for high accuracy detection
JPEG_QUALITY = 82
LATENCY_THRESHOLD_MS = 800

