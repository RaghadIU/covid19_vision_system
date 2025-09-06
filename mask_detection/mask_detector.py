import cv2
from ultralytics import YOLO
import argparse
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True, help="Path to input video")
parser.add_argument("--out", type=str, default="outputs/mask_out.mp4", help="Path to save output video")
parser.add_argument("--model", type=str, default="models/mask_yolov8n.pt", help="Path to mask detection model")
parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--view", action="store_true", help="Show video while processing")
parser.add_argument("--csv", type=str, default="outputs/mask_stats.csv", help="Path to save CSV stats")
args = parser.parse_args()

class MaskDetector:
    def __init__(self, model_path=args.model):
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=args.confidence):
        results = self.model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
        return detections

def draw_box(frame, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    if "no" in label.lower():
        color = (0, 0, 255)
        text = "No Mask"
    else:
        color = (0, 255, 0)
        text = "Mask"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def run_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    detector = MaskDetector(args.model)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    csv_file = open(args.csv, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Frame", "Total_Faces", "Mask", "No_Mask", "Mask_Percentage"])

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = detector.detect(frame)
        mask_count = sum(1 for d in detections if "no" not in d["label"].lower())
        no_mask_count = sum(1 for d in detections if "no" in d["label"].lower())
        total_faces = mask_count + no_mask_count
        mask_percentage = (mask_count / total_faces * 100) if total_faces > 0 else 0

        writer.writerow([frame_idx, total_faces, mask_count, no_mask_count, f"{mask_percentage:.2f}"])

        for det in detections:
            draw_box(frame, det["bbox"], det["label"])

        out.write(frame)
        if args.view:
            cv2.imshow("Mask Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Video saved to: {args.out}")
    print(f"CSV stats saved to: {args.csv}")

if __name__ == "__main__":
    run_video(args.source)
