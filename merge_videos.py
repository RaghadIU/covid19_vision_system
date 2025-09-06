import cv2
import cv2
import os
import numpy as np

output_dir = "C:/Users/HP/Desktop/covid19_vision_sysrem_computervision/outputs"
output_path = os.path.join(output_dir, "final_merged_video.mp4")

video_list = [
    os.path.join(output_dir, "test1_output.mp4"),
    os.path.join(output_dir, "test2_output.mp4"),
    os.path.join(output_dir, "test3_output.mp4"),
]

def merge_videos(video_paths, output_path):
    caps = [cv2.VideoCapture(v) for v in video_paths if os.path.exists(v)]
    if not caps:
        print("No valid videos found.")
        return

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
    print(f"{output_path} saved successfully!")

if __name__ == "__main__":
    merge_videos(video_list, output_path)
