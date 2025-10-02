import os
import shutil
import random

source_folder = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\train"
images_train = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\images\train"
images_val = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\images\val"
labels_train = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\labels\train"
labels_val = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\labels\val"

for folder in [images_train, images_val, labels_train, labels_val]:
    os.makedirs(folder, exist_ok=True)

images = [f for f in os.listdir(source_folder) if f.endswith((".jpg", ".png"))]

random.shuffle(images)

split_ratio = 0.8
split_index = int(len(images) * split_ratio)

for i, img_file in enumerate(images):
    base_name = os.path.splitext(img_file)[0]
    label_file = base_name + ".txt"

    src_img = os.path.join(source_folder, img_file)
    src_label = os.path.join(source_folder, label_file)

    if i < split_index:
        shutil.move(src_img, images_train)
        if os.path.exists(src_label):
            shutil.move(src_label, labels_train)
    else:
        shutil.move(src_img, images_val)
        if os.path.exists(src_label):
            shutil.move(src_label, labels_val)

print("Done")
