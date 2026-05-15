""""
Hard negative mining from COCO dataset.
Downloads airplane images, extracts sky patches, augments them,
and creates empty label files so YOLO treats them as background.
Mind that the coco pictures are color pics, and we want them 3 channels
grayscale with clahe, gaussian blur. Target planes and birds.
"""

import fiftyone.zoo as foz
import cv2
import os
import random
import argparse

N_COCO_SAMPLES = 50
TARGET_SIZE = (1280, 1280)
CROP_TOP_RATIO = 0.4
VAL_SPLIT_PERCENT = 0.2  # 20% to validation

def apply_consistency_pipeline(img):
    """
    Matches UAV positive pipeline: 3 identical channels, clahe, gaussian blur.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    gaussian_blur = cv2.GaussianBlur(gray_clahe, (0, 0), 3)
    # Sharpening
    sharpened = cv2.addWeighted(gray_clahe, 1.5, gaussian_blur, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR) 

def augment_negative(img):
    """
    Simulate lighting variations.
    """
    alpha = random.uniform(0.8, 1.4)
    beta  = random.randint(-20, 40)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def write_empty_label(label_dir: str, img_name: str):
    """
    Creates an empty .txt file for YOLO background images (negative samples).
    """
    os.makedirs(label_dir, exist_ok=True)
    stem = os.path.splitext(img_name)[0]
    with open(os.path.join(label_dir, stem + ".txt"), "w") as f:
        pass

def process_and_save(patch, name, i, total, base_dir):
    """
    Determines if image goes to train or val and saves it.
    """
    split = "train" if i > (total * VAL_SPLIT_PERCENT) else "valid"
    
    img_dir = os.path.join(base_dir, split, "images")
    lbl_dir = os.path.join(base_dir, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    patch = cv2.resize(patch, TARGET_SIZE)
    patch = apply_consistency_pipeline(patch)
    patch = augment_negative(patch)

    out_path = os.path.join(img_dir, name)
    cv2.imwrite(out_path, patch)
    write_empty_label(lbl_dir, name)

def mine_hard_negatives(target_class, output_base):
    print(f"--- Mining {target_class.upper()} from COCO ---")
    
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=[target_class],
        max_samples=N_COCO_SAMPLES,
        shuffle=True,
    )

    count = 0
    num_samples = len(dataset)

    for i, sample in enumerate(dataset):
        img = cv2.imread(sample.filepath)
        if img is None: continue
        h, w = img.shape[:2]

        sky_patch = img[0:int(h * CROP_TOP_RATIO), 0:w]
        process_and_save(sky_patch, f"sky_{target_class}_{sample.id}.jpg", i, num_samples, output_base)

        if "ground_truth" in sample and sample.ground_truth is not None:
            for det_idx, det in enumerate(sample.ground_truth.detections):
                if det.label == target_class:
                    rx1, ry1, rw, rh = det.bounding_box
                    x1, y1 = int(rx1 * w), int(ry1 * h)
                    x2, y2 = int((rx1 + rw) * w), int((ry1 + rh) * h)
                    
                    pad = 50 
                    x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
                    x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
                    
                    obj_patch = img[y1_p:y2_p, x1_p:x2_p]
                    if obj_patch.size > 0:
                        name = f"hard_neg_{target_class}_{sample.id}_{det_idx}.jpg"
                        process_and_save(obj_patch, name, i, num_samples, output_base)
                        count += 1

    print(f"Finished. Mined {num_samples} sky patches and {count} {target_class} object patches.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["bird", "airplane"], default="bird", help="Target class to mine")
    parser.add_argument("--out", default="data/augmented_data", help="Output directory")
    args = parser.parse_args()

    mine_hard_negatives(args.type, args.out)