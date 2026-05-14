# AeroSentry: Robust UAV Perception Pipeline
### AI-Driven Fixed-Wing Detection for Autonomous Interceptor Systems

This repository contains a high-performance detection and tracking pipeline based on **YOLO11n**, optimized for deployment on the **NVIDIA Jetson Orin Nano**. The system is specifically tuned to identify fixed-wing UAVs in challenging sky-region environments with high-velocity motion.

---

## 1. System Architecture
The pipeline employs a multi-stage approach to balance sensitivity and precision:
* **Perception:** YOLO11n (2.6M parameters) for real-time feature extraction.
* **False Positive Reduction:** Class-discriminative training using hard-negative mining (avian distractors).
* **Temporal Logic (Proposed):** Kalman Filter-based state estimation to suppress transient false triggers (clouds/sun glare).

## 2. Dataset & Training
The model was trained on a high-density dataset of fixed-wing signatures, utilizing aggressive augmentation to handle the "jitter" and motion blur inherent in interceptor-mounted footage.

### Dataset Distribution (Post-Augmentation)
* **Total Training Samples:** ~3,400 images
* **Target Class:** `uav` (Fixed-wing drones)
* **Hard Negatives:** Integrated ~300 background samples of avian distractors (birds) to reduce false engagement triggers.

### Training Configuration
| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Model** | `yolo11n.pt` | Optimization for Jetson Orin Nano TensorRT export. |
| **Input Size** | 640px | Trade-off for small-target detection vs. 40+ FPS latency. |
| **Augmentation** | Mosaic, Copy-Paste, HSV | Simulation of varied sky conditions and scale variance. |
| **Optimizer** | SGD (lr0: 0.01) | Robust convergence for binary classification tasks. |

## 3. Results & Evaluation
* **Baseline mAP50:** ~0.85
* **Key Improvement:** Iterative background mining reduced avian False Positives by identifying birds as background noise rather than UAV signatures.

---

## 4. Setup & Deployment

### Installation
```bash
# Clone the repository
git clone [https://github.com/danielluzzatto/uav-detection-pipeline.git](https://github.com/danielluzzatto/uav-detection-pipeline.git)
cd uav-detection-pipeline

# Setup Environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt