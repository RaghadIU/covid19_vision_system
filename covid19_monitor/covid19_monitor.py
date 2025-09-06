import os
import math
import cv2
import pandas as pd
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import numpy as np


output_dir = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\outputs"
os.makedirs(output_dir, exist_ok=True)


human_model_path = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\models\yolov8m.pt"
mask_model_path = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\models\mask_yolov8n.pt"

mask_conf_thresh = 0.6
safe_distance_px = 50


def draw_box(frame, x1, y1, x2, y2, color=(0,0,0), thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def get_center(box):
    x1, y1, x2, y2 = box
    return (int((x1+x2)/2), int((y1+y2)/2))


class HumanDetector:
    def __init__(self, model_name=human_model_path):
        self.model = YOLO(model_name)
    def infer_frame(self, frame):
        results = self.model(frame)[0]
        boxes = []
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls) == 0:  # class 0 = person
                boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        return boxes


class MaskDetector:
    def __init__(self, model_path=mask_model_path):
        self.model = YOLO(model_path)
    def detect(self, face_crop, conf_thresh=mask_conf_thresh):
        results = self.model(face_crop)[0]
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy(),
                                  results.boxes.conf.cpu().numpy()):
            label = self.model.names[int(cls)]
            if conf >= conf_thresh:
                return "Mask" if "Mask" in label else "No Mask"
        return "No Mask"


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{video_name}_output.mp4")
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    human_detector = HumanDetector()
    mask_detector = MaskDetector()
    mtcnn = MTCNN(keep_all=True)

    stats = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        person_boxes = human_detector.infer_frame(frame)
        mask_labels = []

        faces = mtcnn(frame)
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.tolist()]
                face_crop = frame[y1:y2, x1:x2]
                label = mask_detector.detect(face_crop)
                mask_labels.append((x1, y1, x2, y2, label))
                color = (0,255,0) if label=="Mask" else (0,0,255)
                draw_box(frame, x1, y1, x2, y2, color=color, thickness=2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        centers = [get_center(b[:4]) for b in mask_labels]
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0]-centers[j][0]
                dy = centers[i][1]-centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                color = (0,255,0) if dist >= safe_distance_px else (0,0,255)
                cv2.line(frame, centers[i], centers[j], color, 2)

        stats.append({
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "num_people": len(person_boxes),
            "mask_on": sum(1 for l in mask_labels if l[4]=="Mask"),
            "mask_off": sum(1 for l in mask_labels if l[4]=="No Mask")
        })

     
        cv2.imshow("COVID-19 Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    csv_path = os.path.join(output_dir, f"{video_name}_stats.csv")
    pd.DataFrame(stats).to_csv(csv_path, index=False)
    print(f"[OK] Saved video: {out_video_path}")
    print(f"[OK] Saved CSV: {csv_path}")

    return out_video_path


def merge_videos(video_paths, output_path=os.path.join(output_dir,"final_merged_video.mp4")):
    caps = [cv2.VideoCapture(v) for v in video_paths]
    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    fps_list = [c.get(cv2.CAP_PROP_FPS) or 25.0 for c in caps]

    max_width = max(widths)
    max_height = max(heights)
    fps = min(fps_list)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (max_width, max_height))

    for cap in caps:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            scale_w = max_width / w
            scale_h = max_height / h
            scale = min(scale_w, scale_h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            x_offset = (max_width - new_w) // 2
            y_offset = (max_height - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
            writer.write(canvas)
            cv2.imshow("Merged Video", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[OK] Merged video saved at: {output_path}")


if __name__ == "__main__":

    video_list = [
        os.path.join(output_dir, "test1_output.mp4"),
        os.path.join(output_dir, "test2_output.mp4"),
        os.path.join(output_dir, "test3_output.mp4")
    ]

    processed_videos = []
    for v in video_list:
        processed_videos.append(process_video(v))  
    merge_videos(processed_videos)  
