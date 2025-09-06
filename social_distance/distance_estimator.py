import os
import cv2
import math
import csv
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True, help="Path to input video")
parser.add_argument("--out", type=str, default="outputs/distance_out.mp4", help="Path to save output video")
parser.add_argument("--distance_factor", type=float, default=1.5, help="Distance factor for social distancing")
parser.add_argument("--view", action="store_true", help="Show video while processing")
parser.add_argument("--csv", type=str, default="outputs/distances.csv", help="Path to save CSV file")
args = parser.parse_args()

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
    n = len(boxes)
    centers = [get_center(b) for b in boxes]
    heights = [abs(b[3]-b[1]) for b in boxes]
    closest_pairs = []
    for i in range(n):
        min_dist = float('inf')
        closest_j = -1
        for j in range(n):
            if i == j:
                continue
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                closest_j = j
        if closest_j >= 0:
            threshold = max(distance_factor * min(heights[i], heights[closest_j]), 30)
            if min_dist < threshold:
                violations.add(i)
                violations.add(closest_j)
            closest_pairs.append((i, closest_j, min_dist, threshold))
    return closest_pairs, violations

def run_video(source, output_path, distance_factor=1.5, view=False, csv_path=None):
    person_model = YOLO("models/yolov8m.pt")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ensure_dir(os.path.dirname(output_path) or ".")
    ensure_dir(os.path.dirname(csv_path) or ".")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width,height))
    if csv_path:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "person_id", "closest_person_id", "distance_px", "threshold", "violation"])
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        results = person_model(frame)[0]
        boxes = []
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls)==0:
                boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        closest_pairs, violations = check_violations(boxes, distance_factor)
        for idx, b in enumerate(boxes):
            color = (0,0,255) if idx in violations else (0,255,0)
            draw_box(frame, *b, color=color)
        for i, j, dist, threshold in closest_pairs:
            color = (0,255,0) if dist >= threshold else (0,0,255)
            draw_line(frame, get_center(boxes[i]), get_center(boxes[j]), color, f"{dist:.0f}px")
            if csv_path:
                violation = "Yes" if dist < threshold else "No"
                csv_writer.writerow([frame_id, i, j, round(dist,1), round(threshold,1), violation])
        writer.write(frame)
        if view:
            cv2.imshow("Social Distance", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    cap.release()
    writer.release()
    if csv_path:
        csv_file.close()
    if view:
        cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")
    if csv_path:
        print(f"CSV saved to: {csv_path}")

if __name__=="__main__":
    run_video(args.source, args.out, args.distance_factor, args.view, args.csv)
