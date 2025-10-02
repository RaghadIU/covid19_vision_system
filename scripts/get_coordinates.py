import cv2

#We applied this code to all the videos.(test1.mp4,test2.mp4,test3.mp4)
VIDEO_PATH = "videos/test3.mp4"
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", frame)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

cv2.imshow("Select Points", frame)
cv2.setMouseCallback("Select Points", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

with open("video_points.txt", "w") as f:
    for p in points:
        f.write(f"{p[0]},{p[1]}\n")
