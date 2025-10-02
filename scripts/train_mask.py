import os
from ultralytics import YOLO
import argparse

# ======= Arguments =======
parser = argparse.ArgumentParser()
parser.add_argument("--mask_dataset", required=True, help="Path to mask dataset")
parser.add_argument("--output_model", default="models/mask_yolov8n_trained.pt", help="Path to save trained model")
parser.add_argument("--device", default=None, help="Device to use: 'cpu' or '0' for GPU")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--img_size", type=int, default=640, help="Training image size")
parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
args = parser.parse_args()

# ======= Ensure models directory exists =======
os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

# ======= Training =======
def train_mask_model():
    data_yaml = os.path.join(args.mask_dataset, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found in {args.mask_dataset}")
    
    model = YOLO("yolov8n.pt") 
    print("[INFO] Starting training on mask dataset...")
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device
    )
    model.save(args.output_model)
    print(f"[OK] Training finished. Model saved at: {args.output_model}")

if __name__ == "__main__":
    train_mask_model()
