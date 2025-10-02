import os
import shutil
import random



src_mask = r"C:\Users\HP\Downloads\archive (1)\data\with_mask"
src_nomask = r"C:\Users\HP\Downloads\archive (1)\data\without_mask"

dst_train_mask = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\images\train\Mask"
dst_val_mask   = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\images\val\Mask"
dst_train_nomask = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\images\train\NoMask"
dst_val_nomask   = r"C:\Users\HP\Desktop\covid19_vision_sysrem_computervision\mask_dataset\images\val\NoMask"

for folder in [dst_train_mask, dst_val_mask, dst_train_nomask, dst_val_nomask]:
    os.makedirs(folder, exist_ok=True)

def copy_images(src_folder, dst_train, dst_val, num_images=100, train_ratio=0.8):
    files = os.listdir(src_folder)
    random.shuffle(files)
    
    files = files[:num_images]
    
    train_count = int(len(files) * train_ratio)
    
    for i, f in enumerate(files):
        src_path = os.path.join(src_folder, f)
        if i < train_count:
            shutil.copy(src_path, dst_train)
        else:
            shutil.copy(src_path, dst_val)

copy_images(src_mask, dst_train_mask, dst_val_mask, num_images=100)
copy_images(src_nomask, dst_train_nomask, dst_val_nomask, num_images=100)

print(" Done! 100 images per class copied and split into train/val.")
