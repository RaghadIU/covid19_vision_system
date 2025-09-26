from ultralytics import YOLO
import cv2

mask_model = YOLO("C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/models/best.pt")
img = cv2.imread("C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/test1_frame1.jpg")
results = mask_model(img)[0]

for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
    print("Box:", box.cpu().numpy(), "Class:", int(cls))
