import cv2
import os

# قائمة الفيديوهات
VIDEOS = [
    "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/test1.mp4",
    "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/test2.mp4",
    "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/test3.mp4"
]

# الفترة التي تريدها بالثواني لكل فيديو
# يمكن تعديل start_sec و end_sec حسب كل فيديو
TIME_RANGES = [
    (0, 7),   # الفيديو 1: من الثانية 0 إلى 30
    (7, 10),   # الفيديو 2: من الثانية 5 إلى 35
    (4, 6)   # الفيديو 3: من الثانية 10 إلى 40
]

for idx, VIDEO_INPUT in enumerate(VIDEOS):
    start_sec, end_sec = TIME_RANGES[idx]

    cap = cv2.VideoCapture(VIDEO_INPUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # حفظ بنفس الاسم مع _short
    base_name = os.path.splitext(os.path.basename(VIDEO_INPUT))[0]
    VIDEO_OUTPUT = f"C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/videos/{base_name}_short.mp4"

    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret or current_frame > end_frame:
            break
        if current_frame >= start_frame:
            out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"تم قص الفيديو وحفظه: {VIDEO_OUTPUT}")

print("تم الانتهاء من كل الفيديوهات!")
