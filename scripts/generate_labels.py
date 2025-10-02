import os
import cv2
from facenet_pytorch import MTCNN

# ===== إعدادات =====
images_root = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\images"
labels_root = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\labels"

classes = {"Mask": 0, "NoMask": 1}  # class_id

mtcnn = MTCNN(keep_all=True)

def create_yolo_labels(img_path, label_path, class_id):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w, _ = img.shape

    # كشف الوجوه
    boxes, _ = mtcnn.detect(img)

    lines = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            # YOLO format: class_id cx cy w h (normalized 0-1)
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # حفظ الملف
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def process_dataset(split):
    for cls_name, cls_id in classes.items():
        img_dir = os.path.join(images_root, split, cls_name)
        label_dir = os.path.join(labels_root, split, cls_name)

        os.makedirs(label_dir, exist_ok=True)

        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, fname)
                label_path = os.path.join(label_dir, fname.rsplit(".", 1)[0] + ".txt")
                create_yolo_labels(img_path, label_path, cls_id)
                print(f"[OK] {label_path} generated")


if __name__ == "__main__":
    process_dataset("train")
    process_dataset("val")
    print(" All YOLO labels generated successfully!")
