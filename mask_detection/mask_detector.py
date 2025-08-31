import cv2
from ultralytics import YOLO
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True, help="Path to input video")
parser.add_argument("--out", type=str, default="outputs/mask_out.mp4", help="Path to save output video")
parser.add_argument("--view", action="store_true", help="Show video while processing")  
args = parser.parse_args()

source_path = args.source
out_path = args.out

class MaskDetector:
    def __init__(self, model_path="models/mask_yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
        return detections

def draw_box(frame, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    
    label_text = label.split()[0].lower()  

    if "no" in label_text:      
        color = (0, 0, 255)     
        text = "No Mask"
    else:                       
        color = (0, 255, 0)     
        
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def run_video(video_path, mask_model_path="models/mask_yolov8n.pt"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    detector = MaskDetector(mask_model_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        for det in detections:
            draw_box(frame, det["bbox"], f"{det['label']} {det['confidence']:.2f}")

        out.write(frame)
        cv2.imshow("Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    print(f"the video is saved {out_path}")

if __name__ == "__main__":
    video_path = "videos/sample1.mp4"
    run_video(video_path)
