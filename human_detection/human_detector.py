import cv2
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True, help="path to video")
parser.add_argument("--out", type=str, help="path to save output video")
parser.add_argument("--view", action="store_true", help="show video")
args = parser.parse_args()

source_path = args.source
out_path = args.out

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
    inset = 2
    x1 += inset
    y1 += inset
    x2 -= inset
    y2 -= inset
    color = (0, 0, 0)  
    thickness = 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def run_video(video_path, output_path=None, view=True):
    cap = cv2.VideoCapture(video_path)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    detector = HumanDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.infer_frame(frame)
        for bbox in dets:
            draw_box(frame, bbox)

        if view:
            cv2.imshow("Humans Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if output_path:
            out.write(frame)

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video(args.source, args.out, args.view)

