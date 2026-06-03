#!/usr/bin/env python3
"""Live viewer for the GraspVLA cameras (front + side).

Shows each detected camera's scene side by side in one window so you can aim
the BRIOs and tell which is front vs side. Resilient to a camera dropping out
(the flaky side BRIO won't crash the viewer; it shows NO SIGNAL and keeps
trying to reopen).

Usage (needs a display / GUI):
    conda run -n GraspVLA python scripts/show_cameras.py            # auto-detect
    python scripts/show_cameras.py --devices /dev/video4,/dev/video8
    python scripts/show_cameras.py --crop                          # also show the 256 GraspVLA crop

Keys:  q = quit,  s = save a snapshot of every panel,  r = force-reopen all
"""

import argparse
import glob
import os
import re
import subprocess
import time

import cv2
import numpy as np

PANEL_W, PANEL_H = 640, 360          # display size per raw panel (16:9)
TARGET = 256                         # GraspVLA crop side


# color pixel formats we accept (excludes depth 'Z16' and IR 'GREY'/'Y8' nodes)
COLOR_FOURCCS = ("MJPG", "YUYV", "NV12", "UYVY", "RGB3", "BGR3", "YUY2")


def _node_info(dev):
    """Return (card, bus, fourccs) for a /dev/videoN, or None if not a capture node."""
    try:
        info = subprocess.run(["v4l2-ctl", "-d", dev, "--all"],
                              capture_output=True, text=True, timeout=3).stdout
        fmts = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats"],
                              capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None
    if "Video Capture" not in info:
        return None
    card = ""
    bus = dev
    for line in info.splitlines():
        if "Card type" in line:
            card = line.split(":", 1)[1].strip()
        elif "Bus info" in line:
            bus = line.split(":", 1)[1].strip().split()[0]
    fourccs = re.findall(r"'([A-Z0-9 ]+)'", fmts)
    return card, bus, [f.strip() for f in fourccs]


def detect_capture_devices(include_laptop=False):
    """Return the primary COLOR capture node of every EXTERNAL camera.

    Works for Logitech BRIO *and* Intel RealSense (its RGB/color stream is a
    normal UVC node). The laptop's integrated camera is excluded by default.
    Depth ('Z16') and IR ('GREY') nodes are skipped because they carry no color
    format. One node is returned per physical camera (deduped by USB bus).
    """
    per_bus = {}  # bus -> (videoN_int, dev)
    for dev in sorted(glob.glob("/dev/video*"),
                      key=lambda p: int(p.replace("/dev/video", ""))):
        meta = _node_info(dev)
        if meta is None:
            continue
        card, bus, fourccs = meta
        if not include_laptop and "Integrated" in card:
            continue
        if not any(fc in fourccs for fc in COLOR_FOURCCS):
            continue  # depth/IR/metadata node -> skip
        n = int(dev.replace("/dev/video", ""))
        if bus not in per_bus or n < per_bus[bus][0]:
            per_bus[bus] = (n, dev)
    # stable order by node number
    return [dev for _, dev in sorted(per_bus.values())]


def device_label(dev):
    """Human label like 'BRIO (/dev/video5)' or 'RealSense (/dev/video12)'."""
    meta = _node_info(dev)
    card = meta[0] if meta else ""
    if "BRIO" in card:
        name = "BRIO"
    elif "RealSense" in card or "Intel" in card:
        name = "RealSense"
    elif card:
        name = card.split(":")[0]
    else:
        name = "cam"
    return f"{name} ({dev})"


class LiveCam:
    """A camera panel that auto-reopens if it drops."""

    def __init__(self, device, width=1920, height=1080, fps=30):
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self.cap = None
        self.last_open_try = 0.0
        self.open()

    # try configs in order until one delivers a frame: BRIO wants MJPG@1080,
    # RealSense color is YUYV (often capped at 1280x720 over UVC), then fallback
    CONFIGS = [("MJPG", 1920, 1080), ("YUYV", 1280, 720), (None, 640, 480)]

    def open(self):
        self.last_open_try = time.time()
        dev = int(self.device) if str(self.device).isdigit() else self.device
        for fourcc, w, h in self.CONFIGS:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ok, _ = cap.read()  # verify this config actually yields frames
            if ok:
                self.cap = cap
                return
            cap.release()
        self.cap = None

    def read(self):
        """Return a BGR frame, or None if the camera is unavailable."""
        if self.cap is None:
            if time.time() - self.last_open_try > 1.0:  # retry at most 1 Hz
                self.open()
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            self.cap = None
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def graspvla_crop(bgr):
    """Center square crop -> 256x256 (same as scripts/brio_capture.py)."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    sx, sy = w // 2 - side // 2, h // 2 - side // 2
    sq = bgr[sy:sy + side, sx:sx + side]
    return cv2.resize(sq, (TARGET, TARGET), interpolation=cv2.INTER_CUBIC)


def no_signal_panel(label, w, h):
    p = np.zeros((h, w, 3), dtype=np.uint8)
    p[:] = (40, 40, 40)
    cv2.putText(p, "NO SIGNAL", (w // 2 - 90, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(p, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return p


def label_panel(bgr, label, w, h, draw_crop_box=False):
    panel = cv2.resize(bgr, (w, h))
    if draw_crop_box:
        # show the centered square that GraspVLA actually crops
        side = min(bgr.shape[:2])
        scale_x, scale_y = w / bgr.shape[1], h / bgr.shape[0]
        bw, bh = int(side * scale_x), int(side * scale_y)
        x0, y0 = w // 2 - bw // 2, h // 2 - bh // 2
        cv2.rectangle(panel, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 0), 2)
    cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", default=None,
                    help="comma list, e.g. /dev/video4,/dev/video8 (default: auto-detect)")
    ap.add_argument("--labels", default=None,
                    help="comma list of names, e.g. front,side")
    ap.add_argument("--crop", action="store_true",
                    help="also show the 256x256 GraspVLA crop for each camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--outdir", default="brio_out")
    args = ap.parse_args()

    devices = (args.devices.split(",") if args.devices
               else detect_capture_devices())
    if not devices:
        print("No capture-capable cameras found. Plug in a BRIO and retry.")
        return
    labels = (args.labels.split(",") if args.labels
              else [device_label(d) for d in devices])
    while len(labels) < len(devices):
        labels.append(f"cam{len(labels)} ({devices[len(labels)]})")

    print("Showing cameras:")
    for d, l in zip(devices, labels):
        print(f"  {l}: {d}")
    print("Keys: q=quit  s=snapshot  r=reopen")

    cams = [LiveCam(d, args.width, args.height) for d in devices]
    win = "GraspVLA cameras (front | side)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            raw_panels, crop_panels = [], []
            for cam, lab in zip(cams, labels):
                frame = cam.read()
                if frame is None:
                    raw_panels.append(no_signal_panel(lab, PANEL_W, PANEL_H))
                    if args.crop:
                        crop_panels.append(no_signal_panel(lab, TARGET, TARGET))
                else:
                    raw_panels.append(
                        label_panel(frame, lab, PANEL_W, PANEL_H,
                                    draw_crop_box=True))
                    if args.crop:
                        crop = graspvla_crop(frame)
                        crop_panels.append(
                            label_panel(crop, lab + " 256", TARGET, TARGET))

            top = np.hstack(raw_panels)
            if args.crop:
                # pad crop row to the width of the raw row
                crop_row = np.hstack(crop_panels)
                pad = top.shape[1] - crop_row.shape[1]
                if pad > 0:
                    crop_row = np.hstack(
                        [crop_row, np.zeros((crop_row.shape[0], pad, 3), np.uint8)])
                view = np.vstack([top, crop_row])
            else:
                view = top

            cv2.imshow(win, view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("s"):
                os.makedirs(args.outdir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = os.path.join(args.outdir, f"cameras_{ts}.png")
                cv2.imwrite(out, view)
                print("saved", out)
            elif k == ord("r"):
                for cam in cams:
                    cam.release()
                    cam.open()
                print("reopened all cameras")
    finally:
        for cam in cams:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
