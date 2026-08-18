"""
CNN Model Training & Accuracy Fine-Tuning Script
------------------------------------------------
Trains and fine-tunes Convolutional Neural Network (CNN) detection models 
using PyTorch and Ultralytics YOLO engine.

Usage:
    python train.py --epochs 10 --batch-size 16 --imgsz 640 --weights yolo11s.pt
    python train.py --data dataset_example.yaml --weights yolo11m.pt
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO


def create_sample_dataset_config(config_path: str = "dataset_example.yaml") -> str:
    """Generates an example dataset configuration YAML file if none exists."""
    content = """# Dataset Configuration for CNN Object Detection Training
# ----------------------------------------------------
# Directory paths relative to project root:
path: ./dataset      # Dataset root directory
train: images/train  # Training images subdirectory
val: images/val      # Validation images subdirectory
test: images/test    # Optional test images subdirectory

# Class names mapping:
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
"""
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[Dataset Config] Sample dataset configuration created at: {config_path}")
    return config_path


def train_cnn_model(
    weights: str = "yolo11n.pt",
    data_yaml: str = "coco8.yaml",
    epochs: int = 10,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "auto",
    learning_rate: float = 0.01,
    project_name: str = "runs/detect",
    run_name: str = "train_cnn",
):
    """
    Executes deep CNN model training loop, saving best weights to models/best_cnn_model.pt.
    """
    print("=" * 65)
    print("      CNN MODEL ACCURACY TRAINING & FINE-TUNING ENGINE      ")
    print("=" * 65)
    print(f" Base CNN Weights : {weights}")
    print(f" Dataset YAML      : {data_yaml}")
    print(f" Target Resolution : {imgsz}x{imgsz}")
    print(f" Epochs            : {epochs}")
    print(f" Batch Size        : {batch_size}")
    print(f" Learning Rate     : {learning_rate}")
    print(f" Device            : {device}")
    print("=" * 65)

    # Ensure output models directory exists
    os.makedirs("models", exist_ok=True)
    best_weights_target = os.path.abspath(os.path.join("models", "best_cnn_model.pt"))

    # Load base CNN model
    print(f"\n[1/3] Loading PyTorch CNN architecture from base weights: {weights}...")
    model = YOLO(weights)

    # If user passed custom yaml that doesn't exist yet, fallback to sample dataset config
    if data_yaml == "dataset_example.yaml" and not os.path.exists("dataset/images/val"):
        print("\n[Notice] Local 'dataset/images/val' folder is not populated yet.")
        print(" Using 'coco8.yaml' dataset auto-downloader to demonstrate fine-tuning...")
        data_yaml = "coco8.yaml"

    print(f"\n[2/3] Starting CNN training pipeline on dataset '{data_yaml}'...")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device if device != "auto" else "",
            lr0=learning_rate,
            project=project_name,
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        print(f"\n[3/3] Training completed successfully!")

        # Save trained checkpoint to models/best_cnn_model.pt
        best_run_weights = os.path.join(project_name, run_name, "weights", "best.pt")
        if os.path.exists(best_run_weights):
            shutil.copy(best_run_weights, best_weights_target)
            print(f" Saved best trained CNN model to: {best_weights_target}")
        else:
            model.save(best_weights_target)
            print(f" Saved trained CNN weights to: {best_weights_target}")

        print("\n To use your new trained CNN model, set YOLO_MODEL_PATH in app/config.py:")
        print(f'   MODEL_PATH = "{best_weights_target}"\n')

        return results

    except Exception as e:
        print(f"\n[Training Error]: {e}")
        model.save(best_weights_target)
        print(f" Saved baseline CNN weights to: {best_weights_target}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Object Detection Model for High Accuracy")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Base CNN weights (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Path to dataset YAML config (e.g. coco8.yaml, dataset_example.yaml)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution (320, 640, 1280)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, cuda:0, auto)")

    args = parser.parse_args()

    train_cnn_model(
        weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        learning_rate=args.lr,
        device=args.device,
    )
