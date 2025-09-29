import cv2
from ultralytics import YOLO
import numpy as np
import os
import csv

MODEL_DIR = "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/models"

PERSON_MODEL = os.path.join(MODEL_DIR, "yolov8n_person.pt")
FACE_MODEL = os.path.join(MODEL_DIR, "face_yolov8n.pt")
MASK_MODEL = "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/runs/detect/train_new50/weights/last.pt"

DISTANCE_THRESHOLD_CM = 150  

person_model = YOLO(PERSON_MODEL)
face_model = YOLO(FACE_MODEL)
mask_model = YOLO(MASK_MODEL)

VIDEOS = [
    "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/test1.mp4",
    "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/test2.mp4",
    "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/test3.mp4"
]
# Note: We replaced each video with a shorter version (test*_short.mp4) 
# to save processing time. Updated README.md accordingly and attached 
# the corresponding CSV files for each video.

REAL_DIMENSIONS = [(400, 300), (400, 300), (400, 300)]

PTS_SRC_LIST = [
    np.array([[23, 417], [760, 411], [446, 210], [216, 212]], dtype=np.float32),  # v1
    np.array([[4, 418], [765, 408], [413, 159], [301, 156]], dtype=np.float32),   # v2
    np.array([[10, 425], [760, 421], [486, 227], [266, 225]], dtype=np.float32)   # v3
]

def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

SUMMARY_CSV = "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/outputs/summary.csv"
with open(SUMMARY_CSV, mode='w', newline='', encoding='utf-8') as summary_file:
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(["Video", "Distance Violations", "Mask Violations", "Protocol Violators"])

    for idx, VIDEO_INPUT in enumerate(VIDEOS):
        VIDEO_NAME = os.path.basename(VIDEO_INPUT)
        VIDEO_OUTPUT = f"C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/outputs/final_{VIDEO_NAME}"
        CSV_OUTPUT = VIDEO_OUTPUT.replace(".mp4", ".csv")

        cap = cv2.VideoCapture(VIDEO_INPUT)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

        csv_file = open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Person_ID", "Center_X", "Center_Y", "Nearest_Person", "Distance_cm"])

        pts_src = PTS_SRC_LIST[idx]
        real_width_cm, real_height_cm = REAL_DIMENSIONS[idx]
        pixel_width = max(pts_src[:,0]) - min(pts_src[:,0])
        pixel_height = max(pts_src[:,1]) - min(pts_src[:,1])
        scale_x = real_width_cm / pixel_width
        scale_y = real_height_cm / pixel_height
        scale = (scale_x + scale_y) / 2

        frame_idx = 0
        distance_violators = set()
        mask_violators = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            person_results = person_model(frame)[0]
            person_boxes = [box.cpu().numpy() for i, box in enumerate(person_results.boxes.xyxy) if int(person_results.boxes.cls[i])==0]

            face_results = face_model(frame)[0]
            face_boxes = [box.cpu().numpy() for box in face_results.boxes.xyxy]

            mask_results = mask_model(frame)[0]
            mask_boxes = [box.cpu().numpy() for box in mask_results.boxes.xyxy]
            mask_status = [int(cls) for cls in mask_results.boxes.cls]


            for face_id, face_box in enumerate(face_boxes):
                x1, y1, x2, y2 = map(int, face_box)
                is_no_mask = False
                for i, mask_box in enumerate(mask_boxes):
                    if box_iou(face_box, mask_box) > 0.2 and mask_status[i] == 1:
                        is_no_mask = True
                        mask_violators.add(face_id)
                        break
                color = (0,0,255) if is_no_mask else (0,255,0)
                label = "No Mask" if is_no_mask else "Mask"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            centers = []
            for box in person_boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2)/2
                cy = (y1 + y2)/2
                centers.append((cx, cy))


            for i, center_i in enumerate(centers):
                min_dist = float('inf')
                nearest_j = -1
                for j, center_j in enumerate(centers):
                    if i == j:
                        continue
                    dx = center_i[0] - center_j[0]
                    dy = center_i[1] - center_j[1]
                    dist_pixel = np.sqrt(dx**2 + dy**2)
                    dist_cm = dist_pixel * scale
                    if dist_cm < min_dist:
                        min_dist = dist_cm
                        nearest_j = j

                if nearest_j != -1:
                    if min_dist < DISTANCE_THRESHOLD_CM:
                        distance_violators.add(i)

                    x1c = int((person_boxes[i][0]+person_boxes[i][2])/2)
                    y1c = int((person_boxes[i][1]+person_boxes[i][3])/2)
                    x2c = int((person_boxes[nearest_j][0]+person_boxes[nearest_j][2])/2)
                    y2c = int((person_boxes[nearest_j][1]+person_boxes[nearest_j][3])/2)

                    color = (0,255,0) if min_dist >= DISTANCE_THRESHOLD_CM else (0,0,255)
                    cv2.line(frame, (x1c,y1c), (x2c,y2c), color, 2)
                    cv2.putText(frame, f"{int(min_dist)} cm", ((x1c+x2c)//2, (y1c+y2c)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    csv_writer.writerow([frame_idx, i, int(center_i[0]), int(center_i[1]), nearest_j, int(min_dist)])

            cv2.imshow("Mask & Distance Detection", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        protocol_violators = distance_violators.union(mask_violators)
        summary_writer.writerow([VIDEO_NAME, len(distance_violators), len(mask_violators), len(protocol_violators)])
        print(f"{VIDEO_NAME}: Distance={len(distance_violators)}, Mask={len(mask_violators)}, Protocol={len(protocol_violators)}")

        cap.release()
        out.release()
        csv_file.close()

cv2.destroyAllWindows()
