import os
import cv2
import numpy as np
import json
from tqdm import tqdm

# Set paths
DATA_DIR = "data"
IMG_SIZE = 224
X_DIR = "model/X_chunks"
Y_DIR = "model/y_chunks"

# Define label map manually
label_map = {
    "skin_cancer_benign": 0,
    "skin_cancer_malignant": 1,
    "brain_tumor_glioma": 2,
    "brain_tumor_meningioma": 3,
    "brain_tumor_notumor": 4,
    "brain_tumor_pituitary": 5
}

# Save label map
os.makedirs("model", exist_ok=True)
with open("model/label_map.json", "w") as f:
    json.dump(label_map, f)

# Create chunk folders
os.makedirs(X_DIR, exist_ok=True)
os.makedirs(Y_DIR, exist_ok=True)

# Count total images
total_images = 0
for label_name in label_map:
    disease, subfolder = label_name.rsplit("_", 1)
    path = os.path.join(DATA_DIR, disease, "train", subfolder)
    if os.path.exists(path):
        total_images += len(os.listdir(path))

print(f"🔍 Starting streaming preprocessing for {total_images} images...")

# Stream and save
index = 0
with tqdm(total=total_images, desc="📦 Streaming", unit="img") as pbar:
    for label_name, label_id in label_map.items():
        disease, subfolder = label_name.rsplit("_", 1)
        path = os.path.join(DATA_DIR, disease, "train", subfolder)
        print(f"📂 Scanning: {path}")
        if not os.path.exists(path):
            print(f"⚠️ Folder not found: {path}")
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                np.save(os.path.join(X_DIR, f"img_{index:05d}.npy"), img)
                np.save(os.path.join(Y_DIR, f"label_{index:05d}.npy"), np.array(label_id))
                index += 1
            else:
                print(f"⚠️ Could not read image: {img_path}")
            pbar.update(1)

print(f"✅ Streaming complete. Saved {index} image-label pairs to /model/")
