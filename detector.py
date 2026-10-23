"""YOLOv3 weapon detector (OpenCV DNN, CPU).

Usage as a script:
    python detector.py --image photo.jpg
    python detector.py --video clip.mp4
    python detector.py --webcam
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CFG = BASE_DIR / "models" / "yolov3_weapon.cfg"
DEFAULT_WEIGHTS = BASE_DIR / "models" / "yolov3_weapon.weights"
CLASSES = ["weapon"]
BOX_COLOR = (0, 0, 255)  # BGR red


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x, y, w, h


class WeaponDetector:
    def __init__(self, cfg=DEFAULT_CFG, weights=None, conf_threshold=0.5, nms_threshold=0.4,
                 input_size=416):
        weights = Path(weights or os.environ.get("WEAPON_WEIGHTS", DEFAULT_WEIGHTS))
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights}. Train or download YOLOv3 weights for the "
                "'weapon' class and place them there (or set WEAPON_WEIGHTS). See README.")
        self.net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # getUnconnectedOutLayers() returns [[i]] or [i] depending on the OpenCV version
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[int(i) - 1] for i in np.array(self.net.getUnconnectedOutLayers()).flatten()]
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for row in output:
                scores = row[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id] * row[4])  # class prob x objectness
                if confidence < self.conf_threshold:
                    continue
                cx, cy, bw, bh = row[0] * w, row[1] * h, row[2] * w, row[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        return [Detection(CLASSES[class_ids[i]], confidences[i], tuple(boxes[i]))
                for i in np.array(keep).flatten()]


def annotate(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x, y, w, h = d.box
        cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, 2)
        text = f"{d.label} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 6, y), BOX_COLOR, -1)
        cv2.putText(out, text, (x + 3, max(th, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def scan_video(detector: WeaponDetector, path: str, every_n_frames: int = 5, max_hits: int = 12):
    """Run detection on every Nth frame. Returns (hits, frames_checked, fps), where
    hits is a list of (timestamp_seconds, annotated_frame, detections)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hits, index, checked = [], 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every_n_frames == 0:
                checked += 1
                detections = detector.detect(frame)
                if detections and len(hits) < max_hits:
                    hits.append((index / fps, annotate(frame, detections), detections))
            index += 1
    finally:
        cap.release()
    return hits, checked, fps


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Detect weapons in an image, video or webcam feed")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--video")
    src.add_argument("--webcam", action="store_true")
    parser.add_argument("--weights", help="path to YOLOv3 weights (default: models/yolov3_weapon.weights)")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    args = parser.parse_args()

    detector = WeaponDetector(weights=args.weights, conf_threshold=args.conf)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        detections = detector.detect(frame)
        print(f"{len(detections)} weapon(s) detected")
        out = Path(args.image).with_name(Path(args.image).stem + "_detected.jpg")
        cv2.imwrite(str(out), annotate(frame, detections))
        print(f"Annotated image saved to {out}")
        return

    cap = cv2.VideoCapture(0 if args.webcam else args.video)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Weapon detection (q to quit)", annotate(frame, detector.detect(frame)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_cli()
