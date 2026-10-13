"""
Real-time multi-object detection with automatic person tracking from a webcam.

Pipeline:
  1. Detect every object YOLOv8n knows about (the 80 COCO classes: people,
     cups, phones, laptops, backpacks, chairs, bottles, ...) with a single
     `ultralytics.YOLO` inference call per frame.
  2. Among the "person" detections, pick a single "primary" person to follow
     (largest box, or the box that best matches whoever we were already
     tracking) — exactly the same selection logic as before.
  3. Hand that person box to a lightweight single-object tracker (CSRT by
     default) which runs every frame. The tracker interpolates position
     smoothly between detections, which is what keeps the box from
     jittering — re-running full detection on every single frame would give
     a noisier, jumpier box for the person we care most about.
  4. Periodically re-run YOLO detection to correct tracker drift or to
     re-acquire the person if they were lost (e.g. left frame, occluded).
  5. Every non-person detection this frame gets a plain bounding box too —
     no tracker, just whatever YOLO found this frame. Clicking one of those
     boxes shows an info panel (class, confidence, one-line description) in
     the top-right corner; clicking the person box or empty space does not
     open/close that panel in a way that fights step 2-4's automatic flow.

Note on model choice:
  This started out on OpenCV's built-in HOG+SVM person detector (fast to set
  up, zero downloads) but that's a fairly weak, older detector — it
  struggles with non-upright poses, partial occlusion, small/far people, and
  cluttered backgrounds, and it only knows "person", nothing else. YOLOv8n
  via `ultralytics` replaces it: one small (~6MB) model, auto-downloaded on
  first run, that detects all 80 COCO classes with better accuracy and speed
  (especially on a GPU) than HOG+SVM ever gave us.

  Note that "wallet" and "eyeglasses" are NOT among the 80 COCO classes YOLOv8n
  ships with, so they won't be detected out of the box — COCO's closest
  analogues are things like "handbag", "backpack", "suitcase" and "cup",
  "cell phone", "laptop", etc. Detecting wallets/glasses specifically would
  need a custom-trained model.
"""

import argparse
import textwrap
import time

import cv2
from ultralytics import YOLO

# One-line descriptions for every COCO class YOLOv8n can detect. Used to
# populate the click-to-inspect info panel without calling out to any
# external API.
CLASS_DESCRIPTIONS = {
    "person": "A human being.",
    "bicycle": "A pedal-powered two-wheeled vehicle.",
    "car": "A four-wheeled motor vehicle for passengers.",
    "motorcycle": "A two-wheeled motor vehicle.",
    "airplane": "A fixed-wing powered aircraft.",
    "bus": "A large motor vehicle for many passengers.",
    "train": "A series of connected rail vehicles.",
    "truck": "A large motor vehicle for transporting goods.",
    "boat": "A small-to-medium watercraft.",
    "traffic light": "A signal that controls road traffic flow.",
    "fire hydrant": "A street connection point for firefighting water.",
    "stop sign": "A red road sign requiring vehicles to stop.",
    "parking meter": "A coin-operated device for paying parking fees.",
    "bench": "A long seat for multiple people.",
    "bird": "A feathered, winged, egg-laying animal.",
    "cat": "A small domesticated carnivorous mammal.",
    "dog": "A domesticated canine companion animal.",
    "horse": "A large hoofed mammal used for riding/work.",
    "sheep": "A woolly grazing mammal raised for wool/meat.",
    "cow": "A large domesticated bovine raised for milk/meat.",
    "elephant": "A very large mammal with a trunk and tusks.",
    "bear": "A large, heavily built mammal with shaggy fur.",
    "zebra": "An African equine with black-and-white stripes.",
    "giraffe": "A tall African mammal with a very long neck.",
    "backpack": "A bag carried on the back with shoulder straps.",
    "umbrella": "A folding canopy used for rain/sun protection.",
    "handbag": "A small bag for carrying personal items.",
    "tie": "A strip of fabric worn knotted around the collar.",
    "suitcase": "A rectangular case for carrying clothes while traveling.",
    "frisbee": "A flying plastic disc thrown for recreation.",
    "skis": "Long, narrow runners for gliding over snow.",
    "snowboard": "A single board for gliding down snowy slopes.",
    "sports ball": "A ball used in various sports.",
    "kite": "A light frame flown in the wind on a string.",
    "baseball bat": "A smooth wooden or metal club used in baseball.",
    "baseball glove": "A padded leather glove used to catch a baseball.",
    "skateboard": "A short board on four wheels for riding/tricks.",
    "surfboard": "A long board used to ride ocean waves.",
    "tennis racket": "A strung frame used to hit a tennis ball.",
    "bottle": "A narrow-necked container for liquids.",
    "wine glass": "A stemmed glass used for drinking wine.",
    "cup": "A small open container for drinking.",
    "fork": "A utensil with prongs for eating/serving food.",
    "knife": "A bladed utensil for cutting.",
    "spoon": "A utensil with a shallow bowl for eating/serving.",
    "bowl": "A round, deep dish for food or liquid.",
    "banana": "A long curved yellow fruit.",
    "apple": "A round fruit with red, green, or yellow skin.",
    "sandwich": "Food consisting of fillings between bread slices.",
    "orange": "A round citrus fruit with orange skin.",
    "broccoli": "A green vegetable with a tree-like head.",
    "carrot": "An orange root vegetable.",
    "hot dog": "A sausage served in a sliced bun.",
    "pizza": "A baked dish of dough topped with sauce and cheese.",
    "donut": "A ring-shaped fried sweet dough pastry.",
    "cake": "A sweet baked dessert.",
    "chair": "A seat for one person, usually with a back.",
    "couch": "An upholstered seat for multiple people.",
    "potted plant": "A plant grown in a container.",
    "bed": "A piece of furniture for sleeping.",
    "dining table": "A table used for meals.",
    "toilet": "A plumbing fixture for sanitation.",
    "tv": "A screen device for displaying video.",
    "laptop": "A portable, foldable personal computer.",
    "mouse": "A handheld pointing device for a computer.",
    "remote": "A handheld device for controlling electronics.",
    "keyboard": "An input device with keys for typing.",
    "cell phone": "A portable handheld telephone/computer.",
    "microwave": "An appliance that heats food with microwaves.",
    "oven": "An appliance for baking or roasting food.",
    "toaster": "An appliance for toasting bread.",
    "sink": "A basin with a water tap for washing.",
    "refrigerator": "An appliance for keeping food cold.",
    "book": "A set of printed or written pages bound together.",
    "clock": "A device for displaying the time.",
    "vase": "A decorative container, often for flowers.",
    "scissors": "A hand tool with two blades for cutting.",
    "teddy bear": "A soft stuffed toy bear.",
    "hair drier": "A handheld device that blows hot air to dry hair.",
    "toothbrush": "A small brush for cleaning teeth.",
}


def create_tracker(kind: str = "CSRT"):
    """Create a single-object tracker, handling the different places
    OpenCV has stashed the tracker API across versions/builds."""
    kind = kind.upper()

    candidates = []
    if kind == "CSRT":
        candidates = ["TrackerCSRT_create", "legacy.TrackerCSRT_create"]
    elif kind == "KCF":
        candidates = ["TrackerKCF_create", "legacy.TrackerKCF_create"]
    else:
        raise ValueError(f"Unknown tracker kind: {kind}")

    for name in candidates:
        obj = cv2
        ok = True
        for part in name.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj()

    raise RuntimeError(
        "Could not find a CSRT/KCF tracker in this OpenCV build. "
        "Install 'opencv-contrib-python' instead of 'opencv-python'."
    )


def detect_objects(model, frame, conf_threshold=0.4):
    """Run YOLOv8n detection on a frame.

    Returns a list of dicts, one per detection:
      {"box": (x, y, w, h), "cls_name": str, "conf": float}
    covering every COCO class the model supports (people included).
    """
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        detections.append({"box": (x, y, w, h), "cls_name": cls_name, "conf": conf})

    return detections


def iou(box_a, box_b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union


def pick_primary_box(boxes, previous_box):
    """Choose which detected person to track.

    If we already had a tracked box, prefer whichever new detection
    overlaps it the most (keeps tracking the *same* person rather than
    hopping to someone new who happens to be bigger). Otherwise, fall
    back to the largest box, on the assumption the closest/most prominent
    person in frame is the one we want to follow.
    """
    if not boxes:
        return None

    if previous_box is not None:
        best = max(boxes, key=lambda b: iou(b, previous_box))
        if iou(best, previous_box) > 0.1:
            return best

    return max(boxes, key=lambda b: b[2] * b[3])


def point_in_box(x, y, box):
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def make_mouse_callback(state):
    """Click-to-inspect handling for non-person detections.

    - Clicking a non-person box selects it (updates the info panel).
    - Clicking a different non-person box switches the selection.
    - Clicking empty space clears the selection.
    - Clicking the tracked person's box is a no-op: the panel is
      "automatic-only" for the person and must not open or close because
      of a click on them.
    """

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        person_box = state["person_box"]
        if person_box is not None and point_in_box(x, y, person_box):
            return

        hit = None
        hit_area = None
        for det in state["other_detections"]:
            if point_in_box(x, y, det["box"]):
                _, _, w, h = det["box"]
                area = w * h
                if hit is None or area < hit_area:
                    hit = det
                    hit_area = area

        state["selected"] = hit

    return on_mouse


def draw_info_panel(frame, selected):
    if selected is None:
        return

    cls_name = selected["cls_name"]
    conf_pct = f"{selected['conf'] * 100:.1f}%"
    description = CLASS_DESCRIPTIONS.get(cls_name, f"A detected {cls_name}.")

    lines = [f"Object: {cls_name}", f"Confidence: {conf_pct}"]
    lines.extend(textwrap.wrap(description, width=30) or [""])

    frame_h, frame_w = frame.shape[:2]
    padding = 10
    line_h = 20
    panel_w = 260
    panel_h = padding * 2 + line_h * len(lines)

    x0 = frame_w - panel_w - 10
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 200, 255), 1)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x0 + padding, y0 + padding + (i + 1) * line_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Webcam multi-object detection + person tracking")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO weights to use; auto-downloaded on first run (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum detection confidence to keep a box, 0-1 (default: 0.4)",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=15,
        help="Re-run YOLO detection to (re)acquire the tracked person every N frames (default: 15)",
    )
    parser.add_argument(
        "--tracker",
        choices=["CSRT", "KCF"],
        default="CSRT",
        help="Tracker algorithm: CSRT is more accurate, KCF is faster (default: CSRT)",
    )
    parser.add_argument(
        "--lost-frames",
        type=int,
        default=10,
        help="Consecutive failed detections/track-updates before we give up and re-search (default: 10)",
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "See the README for webcam access notes (especially on WSL)."
        )

    window_name = "Object Detector"
    cv2.namedWindow(window_name)

    state = {"person_box": None, "other_detections": [], "selected": None}
    cv2.setMouseCallback(window_name, make_mouse_callback(state))

    tracker = None
    tracking_box = None  # (x, y, w, h) of the currently tracked person
    lost_count = 0
    frame_count = 0

    fps = 0.0
    fps_smoothing = 0.9  # exponential moving average factor
    prev_time = time.time()

    print("Press 'q' to quit. Click a non-person object's box to inspect it.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame_count += 1
            need_detect = tracker is None or frame_count % args.detect_interval == 0

            detections = detect_objects(model, frame, args.conf)
            person_boxes = [d["box"] for d in detections if d["cls_name"] == "person"]
            other_detections = [d for d in detections if d["cls_name"] != "person"]

            if need_detect:
                chosen = pick_primary_box(person_boxes, tracking_box)

                if chosen is not None:
                    tracking_box = tuple(int(v) for v in chosen)
                    tracker = create_tracker(args.tracker)
                    tracker.init(frame, tracking_box)
                    lost_count = 0
                elif tracker is None:
                    # Nobody found yet and nothing to track — keep searching.
                    tracking_box = None
            else:
                ok_track, box = tracker.update(frame)
                if ok_track:
                    tracking_box = tuple(int(v) for v in box)
                    lost_count = 0
                else:
                    lost_count += 1
                    if lost_count > args.lost_frames:
                        # Give up on this person; force a fresh detection
                        # pass (and allow switching targets) next frame.
                        tracker = None
                        tracking_box = None
                        lost_count = 0

            state["person_box"] = tracking_box
            state["other_detections"] = other_detections

            for det in other_detections:
                x, y, w, h = det["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
                cv2.putText(
                    frame,
                    f"{det['cls_name']} {det['conf'] * 100:.0f}%",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 165, 0),
                    2,
                )

            if tracking_box is not None:
                x, y, w, h = tracking_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Person",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Searching for person...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            draw_info_panel(frame, state["selected"])

            now = time.time()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = fps_smoothing * fps + (1 - fps_smoothing) * instant_fps
            prev_time = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
