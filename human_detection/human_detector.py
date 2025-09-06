import cv2
import argparse
import csv
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True, help="path to video")
parser.add_argument("--out", type=str, default="outputs/humans_out.mp4", help="path to save output video")
parser.add_argument("--stats", type=str, default="outputs/humans_stats.csv", help="CSV file for stats")
parser.add_argument("--view", action="store_true", help="show video")
args = parser.parse_args()

class HumanDetector:
    def __init__(self, model_name="models/yolov8m.pt", conf=0.5, iou=0.5, device=None):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.device = device

    def infer_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=[0],
            verbose=False
        )
        dets = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in xyxy:
                dets.append((int(x1), int(y1), int(x2), int(y2)))
        return dets

def draw_box(frame, bbox, label="Human"):
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def run_video(video_path, output_path=None, stats_path=None, view=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h)) if output_path else None

    detector = HumanDetector()
    stats = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        dets = detector.infer_frame(frame)
        count = len(dets)

        for bbox in dets:
            draw_box(frame, bbox)

        cv2.putText(frame, f"People: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        stats.append([frame_idx, count])

        if view:
            cv2.imshow("Humans Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if out:
            out.write(frame)

        if frame_idx % 30 == 0:
            print(f"[INFO] Processed frame {frame_idx}, detected {count} humans.")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Human_Count"])
            writer.writerows(stats)
        print(f"[OK] Stats saved to {stats_path}")

    print("[DONE] Human detection completed.")

if __name__ == "__main__":
    run_video(args.source, args.out, args.stats, args.view)
