import os

folders = [
    "mask_dataset/labels/train/Mask",
    "mask_dataset/labels/train/NoMask",
    "mask_dataset/labels/val/Mask",
    "mask_dataset/labels/val/NoMask"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

print("✅ Labels folders created successfully!")
