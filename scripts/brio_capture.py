#!/usr/bin/env python3
"""BRIO (UVC/OpenCV) capture layer for GraspVLA on Dexmate.

Replaces the controller's RealSense capture (vla_client/utils/cameras.py, which
uses pyrealsense2) with plain OpenCV VideoCapture, and produces the exact image
format GraspVLA's server expects.

GraspVLA image input (see GraspVLA/vla_network/scripts/serve.py:infer_single_sample):
  - two third-person RGB views, keys 'front_view_image' and 'side_view_image'
  - each is a *list* of frames; the server only uses the last one ([-1])
  - each frame is a 256x256x3 uint8 RGB array
  - the model also needs 'proprio_array' (7-D eef pose from robot FK, >=4 steps)
    and 'text' ("pick up {object}"). proprio is NOT vision; for static perception
    testing we send a dummy proprio so we can validate the camera+frame geometry
    without moving the robot.

Crop logic mirrors the original cameras.py exactly: the RealSense delivered
480x640 and was center-cropped to a 480x480 *square* (full vertical FOV, crop
the horizontal) then resized to 256 with INTER_CUBIC. We do the same: take the
largest centered square (full height for a 16:9 frame), then resize to 256.

FOV note: GraspVLA was trained with ~69 H / ~42 V deg cameras. Set the BRIO FOV
preset to 78 deg (-> ~70 H / ~43 V) so a full-height square crop matches. On
Linux the FOV preset is NOT a standard v4l2 control (it is a Logitech UVC
extension, default 90 deg), so set it with Logi Tune on Win/Mac, OR compensate
here with --crop-scale < 1.0 to emulate a narrower vertical FOV in software.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

TARGET = 256  # GraspVLA input side length


class BrioCamera:
    """One UVC camera -> 256x256x3 uint8 RGB frames for GraspVLA."""

    def __init__(self, device, width=1920, height=1080, fps=30,
                 crop_scale=1.0, fourcc="MJPG", warmup=10):
        # device may be an int index or a /dev/videoN path
        self.device = device
        self.crop_scale = float(crop_scale)
        # V4L2 backend is the right one for UVC on Linux
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {device!r}")
        # MJPG is required to get 1080p at full frame rate over USB
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # small buffer so get_frame() returns a fresh frame, not a stale one
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[BrioCamera {device}] opened at {actual_w}x{actual_h} "
              f"(requested {width}x{height}), crop_scale={self.crop_scale}")
        if (actual_w, actual_h) != (width, height):
            print(f"[BrioCamera {device}] WARNING: driver gave {actual_w}x{actual_h}; "
                  f"check the camera supports {width}x{height}")

        for _ in range(warmup):  # let auto-exposure/white-balance settle
            self.cap.read()

    def read_raw_bgr(self):
        """Grab one raw frame as BGR (OpenCV native), no crop/resize."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame from camera {self.device!r}")
        return frame

    @staticmethod
    def crop_and_resize(bgr, crop_scale=1.0):
        """Center square crop (mirrors cameras.py) -> RGB 256x256 uint8.

        crop_scale<1.0 crops a smaller centered square (zoom in) to emulate a
        narrower FOV when the BRIO preset cannot be set to 78 deg.
        """
        h, w = bgr.shape[:2]
        side = int(min(h, w) * crop_scale)
        start_x = w // 2 - side // 2
        start_y = h // 2 - side // 2
        square = bgr[start_y:start_y + side, start_x:start_x + side, :]
        resized = cv2.resize(square, (TARGET, TARGET), interpolation=cv2.INTER_CUBIC)
        # OpenCV is BGR; GraspVLA expects RGB
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def get_frame(self):
        """Processed 256x256x3 uint8 RGB frame, ready for GraspVLA."""
        return self.crop_and_resize(self.read_raw_bgr(), self.crop_scale)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def make_dummy_proprio(n=4):
    """Placeholder eef proprio for static perception tests (robot not moving).

    7-D [x, y, z, roll, pitch, yaw, gripper]. gripper in [-1, 1] (the server
    remaps to [0, 1]); 1.0 = open. The server reads steps [-4] and [-1], so we
    need at least 4 entries.
    """
    step = np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return [step.copy() for _ in range(n)]


def build_graspvla_obs(front_rgb, side_rgb, text="pick up object", proprio=None):
    """Assemble the dict GraspVLA's server expects (see serve.py).

    front_rgb / side_rgb: 256x256x3 uint8 RGB arrays.
    Returns a dict ready to pickle and send over ZMQ.
    """
    if proprio is None:
        proprio = make_dummy_proprio()
    return {
        "text": text,
        "front_view_image": [front_rgb],   # server uses [-1]
        "side_view_image": [side_rgb],
        "proprio_array": proprio,          # server uses [-4] and [-1]
        "compressed": False,
    }


def _validate_obs(obs):
    """Sanity-check an obs dict against GraspVLA's input contract."""
    errs = []
    for key in ("front_view_image", "side_view_image"):
        imgs = obs.get(key)
        if not isinstance(imgs, list) or len(imgs) == 0:
            errs.append(f"{key}: expected non-empty list")
            continue
        img = imgs[-1]
        if not isinstance(img, np.ndarray) or img.shape != (TARGET, TARGET, 3):
            errs.append(f"{key}[-1]: expected ({TARGET},{TARGET},3) array, got "
                        f"{getattr(img, 'shape', type(img))}")
        elif img.dtype != np.uint8:
            errs.append(f"{key}[-1]: expected uint8, got {img.dtype}")
    prop = obs.get("proprio_array")
    if prop is None or len(prop) < 4:
        errs.append("proprio_array: need >=4 steps (server reads [-4] and [-1])")
    elif len(np.asarray(prop[-1])) != 7:
        errs.append(f"proprio_array steps must be 7-D, got {len(np.asarray(prop[-1]))}")
    if not isinstance(obs.get("text"), str) or not obs["text"]:
        errs.append("text: expected non-empty string")
    return errs


# ---------------------------------------------------------------------------
# CLI: preview / snapshot / test (+ optional send to server)
# ---------------------------------------------------------------------------

def _draw_center_cross(bgr, length=50, color=(0, 255, 0)):
    h, w = bgr.shape[:2]
    cv2.line(bgr, (w // 2 - length, h // 2), (w // 2 + length, h // 2), color, 1)
    cv2.line(bgr, (w // 2, h // 2 - length), (w // 2, h // 2 + length), color, 1)


def _panel(cam, label, panel=512):
    """Build a BGR preview panel: processed 256 view scaled up + center cross."""
    rgb = cam.get_frame()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (panel, panel), interpolation=cv2.INTER_NEAREST)
    _draw_center_cross(bgr)
    cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    return bgr


def cmd_preview(front, side, args):
    """Side-by-side live preview of the exact 256x256 frames sent to GraspVLA.

    Eyeball check (README 3.3): front -> robot/workspace roughly centered, table
    level; side -> center cross on workspace center, table edge level.
    """
    print("preview: press 'q' to quit, 's' to save a snapshot pair")
    while True:
        panels = [_panel(front, "FRONT (256->512)")]
        if side is not None:
            panels.append(_panel(side, "SIDE (256->512)"))
        cv2.imshow("GraspVLA BRIO preview", np.hstack(panels))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            _save_pair(front, side, args.outdir)
    cv2.destroyAllWindows()


def _save_pair(front, side, outdir):
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    f = front.get_frame()
    cv2.imwrite(os.path.join(outdir, f"front_{ts}.png"),
                cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    print(f"saved front_{ts}.png  shape={f.shape} dtype={f.dtype}")
    if side is not None:
        s = side.get_frame()
        cv2.imwrite(os.path.join(outdir, f"side_{ts}.png"),
                    cv2.cvtColor(s, cv2.COLOR_RGB2BGR))
        print(f"saved side_{ts}.png  shape={s.shape} dtype={s.dtype}")


def cmd_snapshot(front, side, args):
    _save_pair(front, side, args.outdir)


def cmd_test(front, side, args):
    """Build + validate one GraspVLA obs dict; optionally send it to the server."""
    front_rgb = front.get_frame()
    side_rgb = side.get_frame() if side is not None else front_rgb
    if side is None:
        print("WARNING: only one camera given; duplicating front as side for the test")
    obs = build_graspvla_obs(front_rgb, side_rgb, text=args.text)

    errs = _validate_obs(obs)
    print("\n=== GraspVLA obs dict ===")
    for k, v in obs.items():
        if k in ("front_view_image", "side_view_image"):
            print(f"  {k}: list[{len(v)}] of {v[-1].shape} {v[-1].dtype}")
        elif k == "proprio_array":
            print(f"  {k}: {len(v)} steps x {len(np.asarray(v[-1]))}-D")
        else:
            print(f"  {k}: {v!r}")
    if errs:
        print("\nVALIDATION FAILED:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("\nvalidation OK: obs matches GraspVLA serve.py input contract")

    if args.send:
        _send_to_server(obs, args.server_ip, args.server_port, args.outdir)


def _send_to_server(obs, ip, port, outdir):
    """Send obs to a running GraspVLA server and report action + debug bbox/goal.

    This is the static perception milestone (HANDOFF step 3): no robot motion,
    just verify the camera+coordinate pipeline. bbox should box the object in
    both views; goal + (0,0,0.75) should land on the object in Dexmate base.
    """
    import pickle
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.connect(f"tcp://{ip}:{port}")
    print(f"\nsending obs to GraspVLA server at {ip}:{port} ...")
    sock.send_multipart([b"", pickle.dumps(obs)])
    if sock.poll(30000) == 0:
        print("ERROR: no response within 30s (is the server up?)")
        sys.exit(1)
    parts = sock.recv_multipart()
    reply = pickle.loads(parts[-1])
    print("server reply info:", reply.get("info"))
    action = np.asarray(reply.get("result"))
    print("action sequence shape:", action.shape)
    debug = reply.get("debug", {})
    if "bbox" in debug:
        print("bbox:", debug["bbox"])
    if "pose" in debug:
        goal = np.asarray(debug["pose"])
        print("goal (graspvla base):", goal)
        # README mapping: p_base = p_graspvla + (0,0,0.75)
        if goal.shape and goal.shape[-1] >= 3:
            base_goal = goal.copy()
            base_goal[..., 2] += 0.75
            print("goal (dexmate base, +0.75 z):", base_goal)
    # draw bbox on the views for eyeballing
    _save_bbox_overlay(obs, debug, outdir)


def _save_bbox_overlay(obs, debug, outdir):
    if "bbox" not in debug:
        return
    os.makedirs(outdir, exist_ok=True)
    bbox = debug["bbox"]
    # bbox may be per-view; handle a single box or a list/dict
    for key, view_name in (("front_view_image", "front"), ("side_view_image", "side")):
        img = cv2.cvtColor(obs[key][-1].copy(), cv2.COLOR_RGB2BGR)
        b = bbox.get(view_name) if isinstance(bbox, dict) else bbox
        try:
            b = np.asarray(b).reshape(-1, 4)
            for x0, y0, x1, y1 in b.astype(int):
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        except Exception as e:
            print(f"could not draw bbox on {view_name}: {e}")
        out = os.path.join(outdir, f"bbox_{view_name}.png")
        cv2.imwrite(out, img)
        print("saved", out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["preview", "snapshot", "test"],
                   help="preview: live windows; snapshot: save a pair; "
                        "test: build/validate obs (+--send to a server)")
    p.add_argument("--front", default="/dev/video8",
                   help="front camera index or /dev/videoN (default: /dev/video8)")
    p.add_argument("--side", default=None,
                   help="side camera index or /dev/videoN (optional)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crop-scale", type=float, default=1.0,
                   help="<1.0 zooms in to emulate a narrower FOV if the BRIO "
                        "preset is not 78 deg (e.g. ~0.8 for a 90 deg preset)")
    p.add_argument("--text", default="pick up object",
                   help="instruction sent to GraspVLA")
    p.add_argument("--outdir", default="brio_out")
    p.add_argument("--send", action="store_true",
                   help="(test) send the obs to a running GraspVLA server")
    p.add_argument("--server-ip", default="127.0.0.1")
    p.add_argument("--server-port", default="6666")
    args = p.parse_args()

    def as_device(d):
        if d is None:
            return None
        return int(d) if str(d).isdigit() else d

    front = BrioCamera(as_device(args.front), args.width, args.height,
                       args.fps, args.crop_scale)
    side = None
    if args.side is not None:
        side = BrioCamera(as_device(args.side), args.width, args.height,
                          args.fps, args.crop_scale)

    try:
        {"preview": cmd_preview,
         "snapshot": cmd_snapshot,
         "test": cmd_test}[args.command](front, side, args)
    finally:
        front.release()
        if side is not None:
            side.release()


if __name__ == "__main__":
    main()
