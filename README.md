## UAV Detection Pipeline

Fixed-wing UAV detection and tracking pipeline for interceptor drone onboard cameras.
Built on YOLOv11n + SAHI + CSRT + Kalman Filter. See `report.pdf` for full methodology and results.

---

## Setup

```bash
git clone https://github.com/danielluzzatto/uav-detection-pipeline.git
cd uav-detection-pipeline

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

Download model weights from [Releases](https://github.com/danielluzzatto/uav-detection-pipeline/releases)
and place at `runs/May14-01-21/weights/best.pt`.

---

## Run

```bash
# Basic — run on a video file
python src/main.py --input path/to/video.mp4

# Full options
python src/main.py \
    --input     path/to/video.mp4 \   # input video
    --output    output.mp4        \   # annotated output video
    --json      detections.json   \   # per-frame detections
    --every     5                 \   # run detector every N frames (default: 5)
    --start_frame 0               \   # skip to this frame
    --max_frames  1000            \   # stop after N frames (omit for full video)
    --display                         # show live preview window (press q to quit)
```

Output JSON format (one entry per detected frame):
```json
{
  "frame_id": 75,
  "bbox": [650.23, 357.7, 710.28, 390.55],
  "score": 0.2154,
  "class": "airborne_fixed_wing",
  "source": "detector"            // "detector" | "csrt" | "kalman"
}
```

---


## Results

| Video | FPS | Detection rate | Precision | Recall |
|---|---|---|---|---|
| Demo (1280×720) | 1.5 | 33% | ~100% | ~33% |
| Clip 49 (1920×1080) | 1.3 | 0% | — | — |
| Clip 50 (1920×1080) | 0.6 | 5.4% | 61% | ~4% |

FPS measured on Intel i7-13700H CPU, using pt weights (the Demo video ran at 1.9 fps using onnx weights). Target deployment is Jetson Orin Nano (TensorRT FP16, projected 30-60 FPS after native resolution retraining).

---

## Repo Structure


- src/: Core implementation logic
  - `main.py`: Entry point for the AeroSentry inference pipeline.
  - `get_data.py`: Script for hard-negative mining (COCO integration).
  - `inference_image_check.py`: Validation script for adversarial testing.
  - `dataset_analysis.py`: Statistical breakdown of class imbalances.
- models/: Pre-trained weights for YOLOv11n.
- runs/: Comprehensive training logs, metrics (PR curves, confusion matrices), and best-performing weights for iterations 1–4.
- report/: 
  - `report.pdf`: Full methodology, Jetson Orin Nano optimization strategy, and evaluation.
- requirements.txt: Environment dependencies.
