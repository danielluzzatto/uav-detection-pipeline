# AeroSentry UAV Detection
### Prototype Perception Pipeline for Jetson Orin Nano

This repository contains a YOLOv11-based detection pipeline designed to identify fixed-wing UAVs from an interceptor drone platform.

---

## 1. Dataset Overview
The dataset contains high-density polygon annotations of UAVs.

* **Train:** 2,835 images (2,892 instances)
* **Valid:** 115 images (118 instances)
* **Test:** 159 images (164 instances)
* **Format:** YOLOv11 (1 Class: `uav`)

## 2. Model & Baseline Training
The **YOLOv11n (Nano)** architecture was chosen for its low latency (~3.4ms on T4) and 2.6M parameter efficiency, making it ideal for the **Jetson Orin Nano**.

### Baseline Configuration
| Parameter | Value |
| :--- | :--- |
| **Model** | `yolo11n.pt` |
| **Input Size** | 640x640 |
| **Epochs** | 150 |
| **Optimizer** | SGD (lr0: 0.01, cos_lr: True) |
| **Augmentation** | Mosaic (1.0), Degrees (15.0), HSV (0.4) |

## 3. Results Summary

### Baseline
* **mAP50:** ~0.85
* **Recall:** 0.89 (High sensitivity for interceptor triggers)
* **Inference Speed:** ~3.4ms (T4) / Projected ~40+ FPS (Orin Nano)
* **Key Challenge:** Model currently shows vulnerability to 2D media (posters/billboards) and cloud edges due to a lack of background negative samples.

## 4. Next Steps
* **FP Suppression:** Implement temporal voting logic (detection must persist for >3 frames).
* **Hard Negative Mining:** Add unlabeled images of clouds/birds to the training set.
* **Edge Optimization:** Export to **TensorRT (FP16)** for deployment.

---

## Setup & Inference

```bash
# Clone and Install
git clone [https://github.com/YOUR_USERNAME/aerosentry-uav-detection.git](https://github.com/YOUR_USERNAME/aerosentry-uav-detection.git)
cd aerosentry-uav-detection
pip install ultralytics

# Inference with trained weights
python -c "from ultralytics import YOLO; model = YOLO('models/best.pt'); model.predict(source='data/test/', save=True)"