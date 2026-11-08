#!/usr/bin/env python3
"""
Process video with YOLO + ByteTrack + NVIDIA BodyPose3DNet (ONNX)
"""
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort

# ============================================================
# Configuration
# ============================================================
VIDEO_PATH = r"C:\Users\Welcome\Downloads\SEM 7\projects\3d_posebyme\input\32936-395456369.mp4"

PROJECT_DIR = Path.cwd()
YOLO_MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"
POSE_MODEL_PATH = PROJECT_DIR / "models" / "bodypose3dnet_accuracy.onnx"

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "yolo_nv_3d_bodypose_output.avi"

print("=" * 60)
print("PROCESSING VIDEO: 32936-395456369.mp4")
print("=" * 60)
print(f"Input:  {VIDEO_PATH}")
print(f"Output: {OUTPUT_VIDEO}")
print()

# ============================================================
# File checks
# ============================================================
if not YOLO_MODEL_PATH.exists():
    print(f"❌ YOLO model not found: {YOLO_MODEL_PATH}")
    exit(1)
if not POSE_MODEL_PATH.exists():
    print(f"❌ Pose model not found: {POSE_MODEL_PATH}")
    exit(1)
if not Path(VIDEO_PATH).exists():
    print(f"❌ Input video not found: {VIDEO_PATH}")
    exit(1)

print("✅ All files verified")
print()

# ============================================================
# Load models
# ============================================================
print("[LOADING MODELS]")

# YOLO
model = YOLO(str(YOLO_MODEL_PATH))
model.overrides.update({
    "conf": 0.3,
    "iou": 0.4,
    "classes": [0]  # Person class
})
print("✅ YOLO loaded")

# ONNX BodyPose3DNet
available_providers = ort.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

pose_session = ort.InferenceSession(str(POSE_MODEL_PATH), providers=providers)
print(f"✅ ONNX loaded (providers: {pose_session.get_providers()})")

# Print input info
print("ONNX Inputs:")
for inp in pose_session.get_inputs():
    print(f"  {inp.name}: {inp.shape} {inp.type}")

# ============================================================
# Configuration
# ============================================================
np.random.seed(42)
TRACK_COLORS = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(5000)}
track_history = {}
MAX_TRAIL = 20

COLOR_LEFT = (255, 0, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_CENTER = (0, 255, 0)

SKELETON_CONNECTIONS = [
    (16, 17), (16, 18), (17, 19),
    (0, 3), (3, 6), (6, 15),
    (6, 20), (20, 22), (22, 24),
    (24, 26), (24, 28), (24, 30), (24, 32),
    (6, 21), (21, 23), (23, 25),
    (25, 27), (25, 29), (25, 31), (25, 33),
    (1, 4), (4, 7), (7, 11),
    (11, 13), (13, 7),
    (2, 5), (5, 8), (8, 10), (10, 14), (14, 8),
    (1, 2)
]

def get_color(idx1, idx2):
    left_indices = [1, 4, 7, 10, 13, 20, 22, 24, 26, 28, 30]
    right_indices = [2, 5, 8, 11, 14, 21, 23, 25, 27, 29, 31]
    if idx1 in left_indices or idx2 in left_indices:
        return COLOR_LEFT
    elif idx1 in right_indices or idx2 in right_indices:
        return COLOR_RIGHT
    return COLOR_CENTER

def process_frame(image, model):
    results = model.track(
        image,
        imgsz=1280,
        verbose=False,
        persist=True,
        tracker="bytetrack.yaml"
    )
    detections = []
    if not results:
        return detections
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return detections
    for box, conf, cls, oid in zip(
        boxes.xyxy, boxes.conf, boxes.cls, boxes.id
    ):
        x1, y1, x2, y2 = map(int, box.tolist())
        detections.append({
            "id": int(oid.item()),
            "bbox": (x1, y1, x2, y2),
            "class": results[0].names[int(cls.item())],
            "score": float(conf.item())
        })
    return detections

def get_pose(person_crop):
    img = cv2.resize(person_crop, (192, 256))
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           .astype(np.float32) / 255.0)
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    feed = {
        "input0": img,
        "k_inv": np.zeros((1, 3, 3), dtype=np.float32),
        "t_form_inv": np.zeros((1, 3, 3), dtype=np.float32),
        "scale_normalized_mean_limb_lengths": np.ones(
            (1, 36), dtype=np.float32
        ),
        "mean_limb_lengths": np.zeros(
            (1, 36), dtype=np.float32
        )
    }
    output = pose_session.run(None, feed)[0]
    return output[0]

def draw_pose(img, pts, bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    coords = []
    for p in pts:
        px = int(x1 + (p[0] * w / 192))
        py = int(y1 + (p[1] * h / 256))
        coords.append((px, py))
    for i, j in SKELETON_CONNECTIONS:
        if i >= len(coords) or j >= len(coords):
            continue
        color = get_color(i, j)
        cv2.line(img, coords[i], coords[j], color, 2)
    for px, py in coords:
        cv2.circle(img, (px, py), 2, (255, 255, 255), -1)
        cv2.circle(img, (px, py), 4, (255, 255, 255), 1)
    return img

def draw_annotations(image, detections, padding=5):
    global track_history
    img = image.copy()
    for det in detections:
        tid = det["id"]
        x1, y1, x2, y2 = det["bbox"]
        label = det["class"]
        score = det["score"]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        txt = f"{label} ID:{tid} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(img, txt, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        color = TRACK_COLORS.get(tid, (0, 255, 0))
        cv2.circle(img, (cx, cy), 4, color, -1)
        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append((cx, cy))
        if len(track_history[tid]) > MAX_TRAIL:
            track_history[tid].pop(0)
        pts = track_history[tid]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 2)
    return img

# ============================================================
# Video processing
# ============================================================
print("\n[PROCESSING VIDEO]")
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
print(f"Video FPS: {fps}")

resize_w = 1280
resize_h = 720
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (resize_w, resize_h)
)

if not out.isOpened():
    print("❌ Error: Could not create output video writer")
    cap.release()
    exit(1)

print(f"Output resolution: {resize_w}x{resize_h}")
print("Processing frames...")
print()

track_history = {}
frame_id = 0
pose_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        detections = process_frame(frame, model)
        frame_out = draw_annotations(frame, detections, padding=5)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size > 0:
                try:
                    pts = get_pose(person_crop)
                    frame_out = draw_pose(frame_out, pts, det["bbox"])
                    pose_count += 1
                except Exception as e:
                    print(f"Pose error ID {det['id']}: {e}")

        out.write(frame_out)
        frame_id += 1

        if frame_id % 30 == 0:
            print(f"  Processed {frame_id} frames, pose detections: {pose_count}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Early termination by user")
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ============================================================
# Verify output
# ============================================================
print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
print(f"Total frames processed: {frame_id}")
print(f"Total pose detections: {pose_count}")

if OUTPUT_VIDEO.exists():
    size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)
    print(f"✅ Output video created: {OUTPUT_VIDEO}")
    print(f"   File size: {size_mb:.2f} MB")
else:
    print("❌ Output video was not created")

print("\n" + "=" * 60)
print("DONE - Check outputs folder for the new video")
print("=" * 60)