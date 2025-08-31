import os
import cv2
import math
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True)
parser.add_argument("--out", type=str, default="outputs/distance_out.mp4")
parser.add_argument("--distance_factor", type=float, default=1.5)
parser.add_argument("--mask_model", type=str, default="models/mask_yolov8n.pt")
parser.add_argument("--view", action="store_true")
args = parser.parse_args()

source_path = args.source
out_path = args.out

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_box(frame, x1, y1, x2, y2, color=(0,255,0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def draw_line(frame, pt1, pt2, color=(0,0,255), text=None):
    cv2.line(frame, pt1, pt2, color, 2)
    if text:
        mid = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
        cv2.putText(frame, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_center(box):
    x1, y1, x2, y2 = box
    return (int((x1+x2)/2), int((y1+y2)/2))

def check_violations(boxes, distance_factor):
    violations = set()
    safe_boxes = []
    n = len(boxes)
    centers = [get_center(b) for b in boxes]
    heights = [abs(b[3]-b[1]) for b in boxes]

    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0]-centers[j][0]
            dy = centers[i][1]-centers[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            threshold = max(distance_factor * min(heights[i], heights[j]), 30)
            if dist < threshold:
                violations.add(i)
                violations.add(j)
    safe_boxes = [b for k, b in enumerate(boxes) if k not in violations]
    viol_boxes = [b for k, b in enumerate(boxes) if k in violations]
    return safe_boxes, viol_boxes, centers

def run_video(source, output_path, distance_factor=1.5, mask_model_path="models/mask_yolov8n.pt", view=False):
    # Load YOLO models
    person_model = YOLO("models/yolov8n.pt")
    mask_model = YOLO(mask_model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ensure_dir(os.path.dirname(output_path) or ".")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width,height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        results = person_model(frame)[0]
        boxes = []
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls)==0:  # person class
                boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        safe_boxes, viol_boxes, centers = check_violations(boxes, distance_factor)

        # Draw boxes
        for b in safe_boxes:
            draw_box(frame, *b, color=(0,255,0))
        for b in viol_boxes:
            draw_box(frame, *b, color=(0,0,255))

        n = len(boxes)
        for i in range(n):

            distances = []
            for j in range(n):
                if i==j:
                   continue
                dx = centers[i][0]-centers[j][0]
                dy = centers[i][1]-centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                distances.append((dist, j))
    
  
            distances.sort(key=lambda x: x[0])
            closest = distances[:2] 
    
            for dist, j in closest:
                threshold = max(distance_factor * min(abs(boxes[i][3]-boxes[i][1]), abs(boxes[j][3]-boxes[j][1])), 30)
                color = (0,255,0) if dist>=threshold else (0,0,255)
                draw_line(frame, centers[i], centers[j], color, f"{dist:.0f}px")

    
        writer.write(frame)
        if view:
            cv2.imshow("Social Distance", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

    cap.release()
    writer.release()
    if view:
        cv2.destroyAllWindows()
    return output_path

def main():
    out = run_video(args.source, args.out, args.distance_factor, args.mask_model, view=args.view)
    print(f"Video saved to: {out}")

if __name__=="__main__":
    main()
