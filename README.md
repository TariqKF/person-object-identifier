# Identifier-Bot — Webcam Object Detection & Person Tracking

Detects every object class YOLOv8n knows about (people, cups, phones,
laptops, backpacks, bottles, chairs, and 70+ more COCO classes) in a live
webcam feed. The primary person is automatically tracked frame-to-frame with
a CSRT tracker so their bounding box stays smooth instead of jittering.
Every other detected object gets a plain per-frame box, and clicking one
opens an info panel with its class, confidence, and a short description.
Shows live FPS.

## How it works

- **Detection**: [`ultralytics`](https://github.com/ultralytics/ultralytics)
  YOLOv8n (`yolov8n.pt`), run once per frame. It detects all 80 COCO
  classes, not just people. Weights (~6MB) are downloaded automatically the
  first time you run the app.
- **Person tracking**: among this frame's "person" detections, one primary
  person is picked (see below) and handed to an OpenCV single-object
  tracker (CSRT by default), which updates the box every frame for smooth
  motion. Every `--detect-interval` frames (default 15), YOLO re-detects to
  correct drift or re-acquire the person if lost. This is fully automatic —
  no click needed, and clicking the person's box does nothing (it's not
  part of the click-to-inspect feature).
- **Primary person selection**: if already tracking someone, the detection
  that overlaps the current box the most is preferred (so tracking doesn't
  jump to a different person). Otherwise the largest detected box is
  chosen, assuming that's the closest/most prominent person in frame.
- **Other objects**: every non-person detection gets a plain bounding box
  drawn every frame straight from that frame's YOLO output — no tracker, no
  smoothing, just "what's here right now."
- **Click to inspect**: click inside a non-person object's box to open an
  info panel in the top-right corner showing its class name, detection
  confidence (as a percentage), and a short one-line description pulled
  from a small built-in lookup table (no external API calls). Clicking a
  different object's box updates the panel; clicking empty space clears it.

### Why YOLOv8n instead of HOG+SVM

This project originally used OpenCV's built-in HOG+SVM person detector,
which requires zero downloads but is a fairly weak, older detector: it
struggles with non-upright poses, partial occlusion, small/far people, and
cluttered backgrounds — and it only ever knows "person," nothing else.
YOLOv8n via `ultralytics` replaces it with a small, fast model that detects
all 80 COCO object classes with noticeably better accuracy, at the cost of
one extra dependency and a one-time model download.

**Note:** "wallet" and "eyeglasses" are not among the 80 COCO classes
YOLOv8n ships with, so they won't be detected out of the box. The closest
COCO analogues are things like `handbag`, `backpack`, `suitcase`, `cup`,
`cell phone`, and `laptop`. Detecting wallets/glasses specifically would
require a custom-trained model.

<details>
<summary>Full list of detectable COCO classes</summary>

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush

</details>

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.8+. `opencv-contrib-python` (not the plain `opencv-python`
package) is required because the CSRT/KCF trackers live in the contrib
modules. `ultralytics` pulls in PyTorch, so the first `pip install` will
take a while and use a fair amount of disk space; the first *run* will also
download the `yolov8n.pt` weights file (~6MB) if it isn't already cached
locally.

## Running

```bash
python3 main.py
```

Options:

```bash
python3 main.py --camera 0 --model yolov8n.pt --conf 0.4 --detect-interval 15 --tracker CSRT --lost-frames 10
```

- `--camera`: camera index (default `0`)
- `--model`: Ultralytics YOLO weights to load; auto-downloaded on first run
  (default `yolov8n.pt`)
- `--conf`: minimum detection confidence to keep a box, `0`-`1` (default
  `0.4`)
- `--detect-interval`: frames between YOLO re-detection passes used to
  (re)acquire the tracked person (default `15`)
- `--tracker`: `CSRT` (more accurate, default) or `KCF` (faster, less
  precise) — used only for the tracked person
- `--lost-frames`: consecutive failed tracker updates before giving up and
  re-searching for a person (default `10`)

Click a non-person object's box to see its details in the top-right panel.
Press `q` in the video window to quit.

## Running on WSL / Linux

### Native Linux
Webcam access should just work via `/dev/video0`. If `cv2.VideoCapture(0)`
fails to open, check:
- The device exists: `ls /dev/video*`
- Your user has permission: you may need to be in the `video` group
  (`sudo usermod -aG video $USER`, then log out/in).
- No other application (browser tab, Zoom, etc.) is holding the camera.

### WSL2 (Windows)
WSL2 does not have native USB passthrough, so the webcam isn't visible to
Linux by default, and GUI windows (`cv2.imshow`) need WSLg. Two options:

**Option A — run it on Windows directly (simplest)**
If you're on WSL mainly for a Linux dev environment, it's often easiest to
run this particular script directly with Windows Python instead of inside
WSL, since Windows has native access to the webcam and no passthrough is
needed. `pip install -r requirements.txt` and `python main.py` work the
same way there.

**Option B — pass the webcam through to WSL2**
1. Make sure you're on Windows 11 (or an updated Windows 10) with a recent
   WSL kernel — GUI support (WSLg) and USB/IP support both need this.
2. On Windows, install [usbipd-win](https://github.com/dorssel/usbipd-win)
   (`winget install usbipd`).
3. In an elevated (Administrator) PowerShell:
   ```powershell
   usbipd list
   usbipd bind --busid <BUSID-of-your-webcam>
   usbipd attach --wsl --busid <BUSID-of-your-webcam>
   ```
4. Inside WSL, confirm the device shows up:
   ```bash
   ls /dev/video*
   ```
   If nothing appears, install `v4l-utils` (`sudo apt install v4l-utils`)
   and check `v4l2-ctl --list-devices`. You may also need `sudo usermod -aG
   video $USER` and to restart your WSL session.
5. GUI display: WSLg (bundled with modern WSL2) should let `cv2.imshow`
   open a window automatically. If windows don't appear, run `wsl --update`
   from PowerShell to make sure WSLg is current.

Note: the webcam stays bound to WSL while attached, and Windows apps won't
be able to use it at the same time. Run `usbipd detach --busid <BUSID>`
(back in Windows) to release it.
