"""
MERGE ALL EXTRA DATA INTO EXISTING DATASET
==========================================
Based on your actual folder structure:

Healthy folder has:
  - Healthy nails roboflow test
  - Healthy nails roboflow train
  - Real-time Healthy Dataset

Liver folder has:
  - kaggle test
  - Kaggle train

Psoriasis folder has:
  - Nail Pitting
  - Nail Psoriasis
  - Real time Nail Psoriais Images
  - Test
  - Train
"""

import os
import shutil
import random
from pathlib import Path

# ============================================================
# YOUR EXISTING Train/Val/Test folders (DO NOT CHANGE)
# ============================================================
TRAIN_DIR = r"C:\Users\LOQ\Desktop\Neha Project\Shree Ganesh\Bhuvan dataset\New_dataset\New_dataset\Train"
VAL_DIR   = r"C:\Users\LOQ\Desktop\Neha Project\Shree Ganesh\Bhuvan dataset\New_dataset\New_dataset\Validation"
TEST_DIR  = r"C:\Users\LOQ\Desktop\Neha Project\Shree Ganesh\Bhuvan dataset\New_dataset\New_dataset\Test"

# ============================================================
# ALL EXTRA SOURCE FOLDERS
# ============================================================
BASE = r"C:\Users\LOQ\Desktop\Neha Project\Dataset\Dataset (Healthy, liver and Psoriasis)"

SOURCE_FOLDERS = [
    # HEALTHY
    (rf"{BASE}\Healthy\Healthy nails roboflow test",      "Healthy"),
    (rf"{BASE}\Healthy\Healthy nails roboflow train",     "Healthy"),
    (rf"{BASE}\Healthy\Real-time Healthy Dataset",        "Healthy"),

    # LIVER
    (rf"{BASE}\Liver\kaggle test",                        "Liver_disease"),
    (rf"{BASE}\Liver\Kaggle train",                       "Liver_disease"),

    # PSORIASIS
    (rf"{BASE}\Psoriasis\Nail Pitting",                   "Psoriasis_disease"),
    (rf"{BASE}\Psoriasis\Nail Psoriasis",                 "Psoriasis_disease"),
    (rf"{BASE}\Psoriasis\Real time Nail Psoriais Images", "Psoriasis_disease"),
    (rf"{BASE}\Psoriasis\Test",                           "Psoriasis_disease"),
    (rf"{BASE}\Psoriasis\Train",                          "Psoriasis_disease"),
]

TRAIN_RATIO     = 0.75
VAL_RATIO       = 0.15
RANDOM_SEED     = 42
IMAGE_EXTENSIONS= {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

random.seed(RANDOM_SEED)

def get_all_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))
        images.extend(folder.rglob(f"*{ext.upper()}"))
    return list(set(images))

def count_existing(split_dir, class_name):
    path = Path(split_dir) / class_name
    if not path.exists():
        return 0
    return len([f for f in path.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS])

print("="*60)
print("MERGING ALL EXTRA DATA INTO EXISTING DATASET")
print("="*60)

print("\n📊 EXISTING image counts:")
for cls in ["Healthy", "Liver_disease", "Psoriasis_disease"]:
    t  = count_existing(TRAIN_DIR, cls)
    v  = count_existing(VAL_DIR,   cls)
    te = count_existing(TEST_DIR,  cls)
    print(f"  {cls:20s}: Train={t:4d} | Val={v:4d} | Test={te:4d} | Total={t+v+te:4d}")

class_images = {"Healthy": [], "Liver_disease": [], "Psoriasis_disease": []}

print("\n📂 Scanning source folders...")
for folder_path, class_name in SOURCE_FOLDERS:
    images = get_all_images(folder_path)
    if not images:
        print(f"  ⚠️  No images: {Path(folder_path).name}")
    else:
        class_images[class_name].extend(images)
        print(f"  ✅ {len(images):4d} images ← {Path(folder_path).name} [{class_name}]")

print(f"\n📊 Extra images found per class:")
for cls, imgs in class_images.items():
    print(f"  {cls:20s}: {len(imgs):4d} images")

total_extra = sum(len(v) for v in class_images.values())
if total_extra == 0:
    print("\n❌ No images found! Check folder paths.")
    exit(1)

total_added = {"Train": 0, "Val": 0, "Test": 0}

for class_name, images in class_images.items():
    if not images:
        continue

    random.shuffle(images)
    n       = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "Train": (images[:n_train],              TRAIN_DIR),
        "Val":   (images[n_train:n_train+n_val], VAL_DIR),
        "Test":  (images[n_train+n_val:],        TEST_DIR),
    }

    print(f"\n  {class_name}:")
    for split_name, (split_images, split_dir) in splits.items():
        dest_dir = Path(split_dir) / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        existing_count = count_existing(split_dir, class_name)
        copied = 0

        for i, img_path in enumerate(split_images):
            ext      = img_path.suffix.lower()
            new_name = f"extra_{existing_count+i:05d}{ext}"
            dest     = dest_dir / new_name
            counter  = 0
            while dest.exists():
                counter += 1
                new_name = f"extra_{existing_count+i:05d}_{counter}{ext}"
                dest = dest_dir / new_name
            shutil.copy2(img_path, dest)
            copied += 1

        total_added[split_name] += copied
        print(f"    → {split_name:5s}: +{copied:4d} images")

print("\n" + "="*60)
print("📊 FINAL image counts after merge:")
grand_total = 0
for cls in ["Healthy", "Liver_disease", "Psoriasis_disease"]:
    t  = count_existing(TRAIN_DIR, cls)
    v  = count_existing(VAL_DIR,   cls)
    te = count_existing(TEST_DIR,  cls)
    total = t + v + te
    grand_total += total
    print(f"  {cls:20s}: Train={t:4d} | Val={v:4d} | Test={te:4d} | Total={total:4d}")

print(f"\n  GRAND TOTAL         : {grand_total} images")
print(f"  Added to Train      : +{total_added['Train']}")
print(f"  Added to Val        : +{total_added['Val']}")
print(f"  Added to Test       : +{total_added['Test']}")
print("\n" + "="*60)
print("✅ MERGE COMPLETE! Now run: python train_inceptionv3_v3.py")
print("="*60)