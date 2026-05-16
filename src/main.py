"""
Drone Detection Pipeline
Frame -> Preprocess -> [ROI crop if tracked | full SAHI if not] -> YOLO
     -> FP suppression -> size gate -> motion gate on Kalaman-> Kalman update
     -> CSRT tracker (appearance-based fill between detector frames)
     -> Kalman (fallback if CSRT fails)
     -> output JSON + annotated video
"""

import cv2
import json
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# Config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = 'runs/May14-01-21/weights/best.onnx' #choose onnx or pt
CONF_THRESH = 0.15
SLICE_SIZE = 256
OVERLAP_RATIO = 0.3

INFER_EVERY_N = 5
ROI_SCALE = 2.5
ROI_EXPAND = 2
MIN_AREA_FRAC = 0.0008
MAX_AREA_FRAC = 0.12
MIN_TRACK_AGE = 3
SKY_GATE_Y = 0.65
MIN_ASPECT = 0.3
CSRT_MAX_AGE = 30

FRAME_W, FRAME_H = 1280, 720
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    frame_id: int
    bbox: list
    score: float
    class_name: str = "airborne_fixed_wing"
    source: str = "detector"

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "bbox": self.bbox,
            "score": self.score,
            "class": self.class_name,
            "source": self.source,
        }


# Kalman tracker ────────────────────────────────────────────────────────────
class KalmanTracker:
    def __init__(self, cx: float, cy: float, w: float, h: float):
        self.age = 0
        self.kf = cv2.KalmanFilter(4, 2) # state variables are (positionx, positiony, velocityx, velocityy) and measuring positionx, positiony, so need (4,2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32) # position is updated with x+vx and y+vy, velocity is constant
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32) # track x and y
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3 # penalty on physics model (low means trust)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1 # penalty on sensor value () 
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) # track confidence of the filter, small means certain, gain certainty with hits
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32) # actual position after
        self.mean_w = w
        self.mean_h = h
        self.hits   = 0
        self.misses = 0
        self.age    = 0

    def predict(self) -> tuple[float, float, float, float]:
        "given the current measurements, predict the next x, y"
        pred = self.kf.predict()
        cx = float(np.clip(pred[0, 0], 0, FRAME_W))
        cy = float(np.clip(pred[1, 0], 0, FRAME_H))
        return cx, cy, self.mean_w, self.mean_h

    def update(self, cx: float, cy: float, w: float, h: float, is_detector=False):
        "compare the prediction and the actual position to update"
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        alpha = 0.3
        self.mean_w = alpha * w + (1 - alpha) * self.mean_w
        self.mean_h = alpha * h + (1 - alpha) * self.mean_h
        self.hits += 1
        self.misses = 0  
        if is_detector:
            self.age = 0
            self.last_valid_cx = cx
            self.last_valid_cy = cy
            self.frames_since_valid = 0
        else:
            self.age += 1
            self.frames_since_valid += 1


    def reset_velocity(self):
        """
        Call when re-acquiring after misses.
        Prevents stale velocity from pulling the ROI in the wrong direction.
        """
        self.kf.statePost[2, 0] = 0.0
        self.kf.statePost[3, 0] = 0.0

    def bbox_from_center(self, cx, cy, w, h) -> list:
        return [
            round(max(0, cx - w / 2), 2),
            round(max(0, cy - h / 2), 2),
            round(min(FRAME_W, cx + w / 2), 2),
            round(min(FRAME_H, cy + h / 2), 2),
        ]

    @property
    def confirmed(self) -> bool:
        return self.hits >= MIN_TRACK_AGE


# ── CSRT wrapper ──────────────────────────────────────────────────────────────
class CSRTTracker:
    """
    Initialized on a confirmed detector bbox,
    updated every frame. Falls back to Kalman if update fails.
    """
    def __init__(self):
        self._tracker = None
        self.active   = False
        self.age = 0

    def init(self, frame: np.ndarray, bbox: list):
        """bbox = [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        self._tracker = cv2.TrackerCSRT.create()
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self.active = True
        self.age = 0

    def update(self, frame: np.ndarray) -> Optional[list]:
        """
        Returns [x1, y1, x2, y2] if tracking succeeded, None if lost.
        """
        self.age += 1

        if self.age > CSRT_MAX_AGE:
            #print("  CSRT stale (>30 frames) -> falling back to pure Kalman")
            self.active = False
            return None
        if not self.active or self._tracker is None:
            return None
        ok, (x, y, w, h) = self._tracker.update(frame)
        if not ok or w <= 0 or h <= 0:
            self.active = False
            return None
        x1 = round(max(0, float(x)), 2)
        y1 = round(max(0, float(y)), 2)
        x2 = round(min(FRAME_W, float(x + w)), 2)
        y2 = round(min(FRAME_H, float(y + h)), 2)
        return [x1, y1, x2, y2]

    def reset(self):
        self._tracker = None
        self.active   = False
        self.age = 0


# ── Motion gate ───────────────────────────────────────────────────────────────

def within_size_gate(new_w, new_h, tracker, max_growth=2.5, max_shrink=0.2) -> bool:
    """
    Rejects detections if the area changed too drastically compared to the 
    running mean tracked by Kalman. 2.5 more means the side grew by sqrt(2.5) ~ 1.6, shrinked by sqrt(0.2) ~ 0.45 
    Use tracker to get the mean size, which is updated with a moving averafe
    """
    current_area = tracker.mean_w * tracker.mean_h
    new_area = new_w * new_h
    
    growth_ratio = new_area / current_area
    
    if growth_ratio > max_growth or growth_ratio < max_shrink:
        #print(f"    size gate: ratio={growth_ratio:.2f} rejected")
        return False
    return True

def within_motion_gate(new_box, prev_cx, prev_cy, frames_elapsed, 
                       img_w, box_scale_factor=1.5) -> bool:
    """
    Modular motion gate that scales based on the size of the detection.
    Allows for larger jumps when the target is closer (larger box).
    """
    new_cx = (new_box[0] + new_box[2]) / 2
    new_cy = (new_box[1] + new_box[3]) / 2
    
    box_w = new_box[2] - new_box[0]
    box_h = new_box[3] - new_box[1]
    drone_size = max(box_w, box_h)
    
    # Logic: max jump = (Size of Drone) * (Velocity Factor) * (Time)
    # Plus a small constant floor (e.g., 1% of screen) to handle very small specks
    base_limit = img_w * 0.01 
    dynamic_limit = (drone_size * box_scale_factor) + base_limit
    
    max_allowed = dynamic_limit * frames_elapsed
    
    dist = np.sqrt((new_cx - prev_cx)**2 + (new_cy - prev_cy)**2)
    
    if dist > max_allowed:
        #print(f"    motion gate: dist={dist:.1f} > max={max_allowed:.1f} (size={drone_size:.1f}) → rejected")
        return False
    return True

# ── Models ────────────────────────────────────────────────────────────────────
def load_models():
    """
    The model was trained on (640,640) pics and we run infernce on (1280, 1280). Since the drones are very small
    compared to the ones seen in training, we use sahi, which divides the image in multiple subimages, and YOLO is 
    run on each of them. Run it on CPU 
    """
    yolo = YOLO(WEIGHTS_PATH)
    sahi = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=WEIGHTS_PATH,
        confidence_threshold=CONF_THRESH,
        device="cpu",
    )
    return yolo, sahi


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img: np.ndarray) -> np.ndarray:
    """
    The model was trained on 3 equal channels (grayscale) images, using clahe and gaussian blur to make the 
    features pop. Do the same at inference
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def is_valid_detection(x1, y1, x2, y2) -> bool:
    """
    make sure image is within bounds, of the right size, and right aspect (for a drone we expect the width to be larger than the height)
    """
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    area_frac = (w * h) / (FRAME_W * FRAME_H)
    if not (MIN_AREA_FRAC < area_frac < MAX_AREA_FRAC):
        return False
    if (y1 + y2) / 2 / FRAME_H > SKY_GATE_Y:
        return False
    if (w / h) < MIN_ASPECT:
        return False
    return True


# ── Inference ─────────────────────────────────────────────────────────────────
def roi_inference(yolo, frame_pre, cx, cy, w, h, scale=ROI_SCALE):
    """
    Keep track of the region of interest (ROI). It can be max the frame size, and usually is the current height and width multiplied by a factor
    scale. This ensures the model looks at the correct area (assume that the drone stays in the same position more or less)
    Assume there is only one drone per frame, and keep the prediction with highest confidence, granted that it is within the bounds
    """
    roi_w = min(FRAME_W, w * scale)
    roi_h = min(FRAME_H, h * scale)
    x1 = int(max(0, cx - roi_w / 2))
    y1 = int(max(0, cy - roi_h / 2))
    x2 = int(min(FRAME_W, cx + roi_w / 2))
    y2 = int(min(FRAME_H, cy + roi_h / 2))

    crop = frame_pre[y1:y2, x1:x2]
    results = yolo(crop, conf=CONF_THRESH, iou=0.3, imgsz=640, verbose=False)[0]

    # predict only a bbox per frame. Assume there is only one uav
    best_conf, best_box = 0, None
    for box in results.boxes:
        if int(box.cls.item()) != 1:  # 1 = uav
            continue
        conf = box.conf.item()
        if conf <= best_conf:
            continue
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        fx1, fy1 = bx1 + x1, by1 + y1
        fx2, fy2 = bx2 + x1, by2 + y1
        if is_valid_detection(fx1, fy1, fx2, fy2):
            best_conf = conf
            best_box = [round(fx1,2), round(fy1,2), round(fx2,2), round(fy2,2)]

    return best_box, best_conf


def full_frame_inference(sahi, frame_pre, tmp="/tmp/sahi_in.jpg"):
    """
    When the roi is not reliable, use sahi to detect the full frame
    """
    cv2.imwrite(tmp, frame_pre)
    preds = get_sliced_prediction(
        tmp, sahi,
        slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO, overlap_width_ratio=OVERLAP_RATIO,
        verbose=0,
    ).object_prediction_list

    best_conf, best_box = 0, None
    for p in preds:
        if p.category.name != 'uav':  # reject 'none' class
            continue
        x1, y1, x2, y2 = p.bbox.to_xyxy()
        if not is_valid_detection(x1, y1, x2, y2):
            continue
        if p.score.value > best_conf:
            best_conf = p.score.value
            best_box  = [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]

    return best_box, best_conf


# ── Visualization ─────────────────────────────────────────────────────────────
COLOR = {
    "detector": (0, 255,   0),   # green
    "csrt":     (255, 165, 0),   # blue
    "kalman":   (0,  165, 255),  # orange
}

def draw(frame: np.ndarray, det: Detection) -> np.ndarray:
    out = frame.copy()
    x1,y1,x2,y2 = [int(c) for c in det.bbox]
    color = COLOR.get(det.source, (255,255,255))
    cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
    if det.source == "detector":
        cv2.putText(out, f"UAV {det.score:.2f} [{det.source}]",
                    (x1, max(y1-8,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    else:
        cv2.putText(out, f"UAV {det.source}",
                    (x1, max(y1-8,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(video_path: str, output_video_path: str = "output.mp4", output_json_path:  str = "detections.json", infer_every: int = INFER_EVERY_N, max_frames: Optional[int] = None, start_frame: Optional[int] = 0, display: bool = False):
    yolo, sahi = load_models()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        #print(f"Error: could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (FRAME_W, FRAME_H))

    kalman: Optional[KalmanTracker] = None
    csrt = CSRTTracker()
    all_dets: list[dict] = []
    frame_id = 0
    det_count = 0
    current_infer_every = infer_every
    detection_type = {"ROI": 0, "ROI x2": 0, "ROI x4": 0, "full_frame_search": 0, 'full_frame_fallback': 0}
    start = time.time()

    print(f"Video: {FRAME_W}x{FRAME_H} @ {fps:.1f}fps — {total} frames")
    print(f"Inference every {infer_every} frames \n")

    while True:
        ret, frame = cap.read()
        # dev: use only some frames for inspection 
        if not ret or (max_frames and frame_id >= max_frames):
            break
        frame_id += 1
        if frame_id < start_frame:
            continue
        processed = preprocess(frame)
        det: Optional[Detection] = None

        # ── Inference frame: detection ───────────────────────────────────────────────────
        if frame_id % current_infer_every == 0:
            box, conf = None, 0.0

            if kalman is not None:
                cx, cy, w, h = kalman.predict()
                
                # First try to get something within the roi
                box, conf = roi_inference(yolo, processed, cx, cy, w, h)

                # If there is nothing in the ROI, expand it by 2x and check there
                if box is not None:
                    detection_type["ROI"] +=1
                else:
                    box, conf = roi_inference(yolo, processed, cx, cy, w * 2, h * 2, scale=1.0)
                    # if still none, expand by 4
                    if box is not None:
                        detection_type["ROI x2"] +=1
                    else:
                        box, conf = roi_inference(yolo, processed, cx, cy, w * 4, h * 4, scale=1.0)
                    # Otherwise fallback on the full frame checking (sahi and so on)
                        if box is not None: 
                            detection_type["ROI x4"] +=1
                        else:
                            #print(f"  Frame {frame_id:05d}: ROI miss → full-frame fallback")
                            box, conf = full_frame_inference(sahi, processed)
                            detection_type["full_frame_fallback"] +=1

                # If there is a box but it is of a bad size (too big or too small) discard it
                if box is not None:
                    bw, bh = box[2] - box[0], box[3] - box[1]
                    if not within_size_gate(bw, bh, kalman):
                        #print(f"  Frame {frame_id:05d}: Size gate rejection")
                        box = None 
                
                if box is not None and kalman.confirmed:
                    h, w = frame.shape[:2]
                    frames_elapsed = kalman.frames_since_valid + current_infer_every
                    if not within_motion_gate(box, kalman.last_valid_cx, kalman.last_valid_cy, frames_elapsed, img_w=w):
                        box = None

                # If there is no box increase kalman misses, if they r more than 3 reset kalman filter and tracker
                if box is None:
                    kalman.misses += 1
                    if kalman.misses >= 3:
                        #print(f"  Frame {frame_id:05d}: track lost")
                        kalman = None
                        csrt.reset()
                        current_infer_every = infer_every
                else:
                    kalman.misses = 0 # Reset miss counter on a successful gate-passed hit

            else: 
                # No track active: standard full-frame search
                box, conf = full_frame_inference(sahi, processed)
                detection_type["full_frame_search"] += 1
            # update Kalman + reinit CSRT on every accepted detection
            if box is not None:
                x1,y1,x2,y2 = box
                bw, bh = x2-x1, y2-y1
                cx, cy = (x1+x2)/2, (y1+y2)/2
                was_missing = kalman is not None and kalman.misses > 0
                if kalman is not None:
                    if was_missing:
                        kalman.reset_velocity()
                    kalman.update(cx, cy, bw, bh, is_detector=True)

                else:
                    kalman = KalmanTracker(cx, cy, bw, bh)

                # reinit CSRT with the original (non-preprocessed) frame
                csrt.init(frame, box)
                
                # usually if confidence is low it means the target is far, so do more frequent check
                infer_every = 2 if conf < 0.20 else infer_every

                if kalman.confirmed:
                    det = Detection(frame_id, box, round(conf,4), source="detector")
                    #print(f"  Frame {frame_id:05d} | conf={conf:.3f} | "
                        #   f"bbox={box} | hits={kalman.hits}")

        # ── Non-inference frame: no detection, tracking and kalman ───────────────────────────────────────────────
        else:
            if kalman is not None and kalman.confirmed:

                # CSRT is the first thing we try
                csrt_box = csrt.update(frame)
                if csrt_box is not None and is_valid_detection(*csrt_box):
                    cx = (csrt_box[0] + csrt_box[2]) / 2
                    cy = (csrt_box[1] + csrt_box[3]) / 2
                    # update Kalman so it stays aligned
                    kalman.update(cx, cy,
                                  csrt_box[2]-csrt_box[0],
                                  csrt_box[3]-csrt_box[1], is_detector=False)
                    det = Detection(frame_id, csrt_box, score=0.0, source="csrt")

                else:
                    # CSRT failed, fall back to Kalman prediction
                    cx, cy, w, h = kalman.predict()
                    box = kalman.bbox_from_center(cx, cy, w, h)
                    det = Detection(frame_id, box, score=0.0, source="kalman")
        
        # expire after 30 frames with no detections
        if kalman is not None and kalman.age > 30:
            #print(f"  Frame {frame_id:05d}: track expired (age={kalman.age})")
            kalman = None
            csrt.reset()
            current_infer_every = infer_every
            

        # ── Output ────────────────────────────────────────────────────────────
        
        if det:
            all_dets.append(det.to_dict())
            det_count += 1
            frame = draw(frame, det)

        if display: 
            status = "TRACKING" if kalman else "SEARCHING"
            cv2.putText(frame, f"Frame: {frame_id} | {status}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            writer.write(frame)
            cv2.imshow("UAV Detection", frame)
            
            # Press 'q' to stop processing and save
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #print("Interrupted by user.")
                break
        

        if frame_id % 200 == 0:
            elapsed = time.time() - start
            src = f"active(hits={kalman.hits})" if kalman else "lost"
            #print(f"[{frame_id}/{total}] {frame_id/elapsed:.1f} fps | "
                #   f"{det_count} dets | kalman={src} | csrt={csrt.active}")

    cap.release()
    writer.release()

    elapsed = time.time() - start
    print(f"\nDone. {frame_id} frames in {elapsed:.1f}s ({frame_id/elapsed:.1f} fps)")
    print(f"Detections: {det_count} / {frame_id} ({100*det_count/max(frame_id,1):.1f}%)")

    with open(output_json_path, 'w') as f:
        json.dump(all_dets, f, indent=2)
    with open('detection_type.json', 'w') as f:
        json.dump(detection_type, f, indent=2)
    #print(f"Saved: {output_video_path}  {output_json_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default="data/videos/Arsuf F1 09_04_2025.mp4")
    p.add_argument('--output', default='output.mp4')
    p.add_argument('--json', default='detections.json')
    p.add_argument('--every', type=int, default=INFER_EVERY_N)
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--max_frames', type=int, default=None)
    p.add_argument('--display', action='store_true', help="Show the video stream during processing")
    args = p.parse_args()

    run_pipeline(
        video_path=args.input,
        output_video_path=args.output,
        output_json_path=args.json,
        infer_every=args.every,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        display = args.display
    )