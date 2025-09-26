import sys, os, math, cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mask_detection.mask_detector import MaskDetector
from human_detection.human_detector import HumanDetector

# ----------- إعداد المتغيرات ----------------
class Args:
    source = r"C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/sample1.mp4"
    out = r"C:/Users/HP/Desktop/outputs/output_video.mp4"
    human_model = "models/yolov8m.pt"
    mask_model = r"C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/models/best.pt"
    device = None
    view = True

args = Args()

# ---------- دوال مساعدة ----------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_box(frame, x1, y1, x2, y2, color=(0,0,0), thickness=1):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

class SocialDistanceEstimator:
    def get_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2)/2), int((y1 + y2)/2)

# ---------- تشغيل الفيديو ----------
def run_video(source, output_path, human_model, mask_model_path, device=None, view=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ensure_dir(os.path.dirname(output_path) or ".")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))

    human_detector = HumanDetector(model_name=human_model, device=device)
    mask_detector = MaskDetector(model_path=mask_model_path)
    distance_estimator = SocialDistanceEstimator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        person_boxes = human_detector.infer_frame(frame)
        face_boxes = []
        masks = []

        # كشف الوجه وكمامة
        for (x1, y1, x2, y2) in person_boxes:
            # تقريب الوجه داخل الصندوق
            face_x1 = x1 + int((x2-x1)*0.15)
            face_x2 = x2 - int((x2-x1)*0.15)
            face_y1 = y1
            face_y2 = y1 + int((y2-y1)*0.6)
            face_boxes.append((face_x1, face_y1, face_x2, face_y2))

            face_crop = frame[face_y1:face_y2, face_x1:face_x2]
            mask_result = mask_detector.detect(face_crop)

            # تحديد الكمامة
            if mask_result and len(mask_result) > 0:
                label = mask_result[0].get("label", "").lower()
                if "no mask" in label:
                    mask_label = "No Mask"
                else:
                    mask_label = "Mask"
            else:
                mask_label = "No Mask"
            masks.append(mask_label)

        # حساب المسافات بين الأشخاص
        centers = [distance_estimator.get_center(b) for b in face_boxes]
        closest_distances = []
        for i, c1 in enumerate(centers):
            min_dist = float('inf')
            for j, c2 in enumerate(centers):
                if i == j:
                    continue
                dx = c1[0]-c2[0]
                dy = c1[1]-c2[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
            closest_distances.append(min_dist)

        # رسم الصناديق والمعلومات
        for i, (box, mask_label) in enumerate(zip(face_boxes, masks)):
            x1, y1, x2, y2 = box
            color = (0,0,255) if mask_label.lower() == "no mask" else (0,255,0)
            draw_box(frame, x1, y1, x2, y2, color=color, thickness=2)

            safe_distance_threshold = 50  # البكسل
            if closest_distances[i] < safe_distance_threshold:
                risk = "High Risk"
                risk_color = (0,0,255)
            else:
                risk = "Safe"
                risk_color = (0,255,0)

            # overlay info
            info_x1 = x2 + 5
            info_y1 = y1
            info_x2 = x2 + 120
            info_y2 = y1 + 60
            overlay = frame.copy()
            cv2.rectangle(overlay, (info_x1, info_y1), (info_x2, info_y2), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # مربعات صغيرة للكمامة والمخاطرة
            cv2.rectangle(frame, (info_x1+5, info_y1+5), (info_x1+20, info_y1+20),
                          (0,0,255) if mask_label.lower()=="no mask" else (0,255,0), -1)
            cv2.rectangle(frame, (info_x1+5, info_y1+45), (info_x1+20, info_y1+60), risk_color, -1)

            # نصوص المعلومات
            cv2.putText(frame, f"Mask: {mask_label}", (info_x1+25, info_y1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
            cv2.putText(frame, f"Dist: {closest_distances[i]:.1f}px", (info_x1+25, info_y1+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
            cv2.putText(frame, f"Risk: {risk}", (info_x1+25, info_y1+55),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

        writer.write(frame)
        if view:
            cv2.imshow("COVID-19 Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if view:
        cv2.destroyAllWindows()
    return output_path

def main():
    out = run_video(args.source, args.out,
                    args.human_model, args.mask_model, args.device, args.view)
    print(f"[OK] Saved: {out}")

if __name__=="__main__":
    main()
