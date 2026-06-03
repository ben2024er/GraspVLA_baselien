#!/usr/bin/env python3
"""Capture the two GraspVLA RGB views (front + side) as 256x256x3 uint8 RGB.

This is the image half of GraspVLA's input. Per serve.py the model wants two
third-person RGB views, each a 256x256x3 uint8 RGB array, supplied as
'front_view_image' and 'side_view_image' (lists; the server uses [-1]).

Crop matches the original RealSense pipeline (vla_client/utils/cameras.py):
center square crop (full vertical FOV, crop horizontal) -> resize 256 (cubic).

Defaults to the user's setup: front=/dev/video5, side=/dev/video9.

Examples:
    # save front_view.png + side_view.png and print shapes/dtype
    conda run -n GraspVLA python scripts/capture_views.py

    # also dump a ready-to-send .npz (front + side arrays)
    conda run -n GraspVLA python scripts/capture_views.py --npz views.npz

    # use as a library:
    from capture_views import get_views
    front_rgb, side_rgb = get_views()          # two (256,256,3) uint8 RGB arrays
"""

import argparse
import os
import pickle

import cv2
import numpy as np

TARGET = 256  # GraspVLA input side length

# capture configs tried in order: BRIO -> MJPG@1080; RealSense color -> YUYV@720
_CONFIGS = [("MJPG", 1920, 1080), ("YUYV", 1280, 720), (None, 640, 480)]


def open_camera(device, fps=30):
    """Open a UVC camera, trying configs until one delivers a frame."""
    dev = int(device) if str(device).isdigit() else device
    for fourcc, w, h in _CONFIGS:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    raise RuntimeError(f"could not open camera {device!r} with any config")


def to_graspvla_rgb(bgr):
    """OpenCV BGR frame -> 256x256x3 uint8 RGB (center square crop + resize)."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    sx, sy = w // 2 - side // 2, h // 2 - side // 2
    square = bgr[sy:sy + side, sx:sx + side]
    resized = cv2.resize(square, (TARGET, TARGET), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def grab(cap, warmup=10):
    """Flush a few frames (auto-exposure) then return one 256x256x3 RGB frame."""
    frame = None
    for _ in range(max(1, warmup)):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("failed to read frame from camera")
    return to_graspvla_rgb(frame)


def get_views(front="/dev/video5", side="/dev/video9", warmup=10):
    """Return (front_rgb, side_rgb): two (256,256,3) uint8 RGB arrays."""
    fcap = open_camera(front)
    scap = open_camera(side)
    try:
        return grab(fcap, warmup), grab(scap, warmup)
    finally:
        fcap.release()
        scap.release()


def build_obs(front_rgb, side_rgb, text="pick up object", proprio=None):
    """Wrap the two views into GraspVLA's serve.py input dict (image half).

    proprio (7-D eef pose x>=4 steps) comes from the robot FK; a dummy open-
    gripper proprio is filled in for static perception tests if not provided.
    """
    if proprio is None:
        step = np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        proprio = [step.copy() for _ in range(4)]
    return {
        "text": text,
        "front_view_image": [front_rgb],   # server uses [-1]
        "side_view_image": [side_rgb],
        "proprio_array": proprio,
        "compressed": False,
    }


def save_bundle(outdir, front_rgb, side_rgb, proprio_arr, text):
    """Write the 3 GraspVLA inputs into one folder; return the folder path.

    Files:
      front_view.png / side_view.png : the two 256x256x3 RGB views
      proprio_array.npy              : (steps, 7) float32 eef pose + gripper
      text.txt                       : the "pick up {object}" instruction
      obs.pkl                        : the full pickled dict serve.py expects
                                       (front_view_image/side_view_image lists,
                                        proprio_array, text, compressed=False)
    """
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, "front_view.png"),
                cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(outdir, "side_view.png"),
                cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(outdir, "proprio_array.npy"), proprio_arr)
    with open(os.path.join(outdir, "text.txt"), "w") as f:
        f.write(text + "\n")
    obs = {
        "text": text,
        "front_view_image": [front_rgb],
        "side_view_image": [side_rgb],
        "proprio_array": [row for row in proprio_arr],
        "compressed": False,
    }
    with open(os.path.join(outdir, "obs.pkl"), "wb") as f:
        pickle.dump(obs, f)
    return outdir


def load_bundle(outdir):
    """Load a saved bundle back into the GraspVLA obs dict (for sending later)."""
    with open(os.path.join(outdir, "obs.pkl"), "rb") as f:
        return pickle.load(f)


def _check(name, img):
    ok = (isinstance(img, np.ndarray) and img.shape == (TARGET, TARGET, 3)
          and img.dtype == np.uint8)
    print(f"  {name}: shape={img.shape} dtype={img.dtype} "
          f"{'OK' if ok else 'BAD -- expected (256,256,3) uint8'}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--front", default="/dev/video5", help="front camera (default /dev/video5)")
    ap.add_argument("--side", default="/dev/video9", help="side camera (default /dev/video9)")
    ap.add_argument("--outdir", default="brio_out")
    ap.add_argument("--npz", default=None, help="also save both arrays to this .npz")
    ap.add_argument("--object", default=None,
                    help="object name -> instruction becomes 'pick up {object}'")
    ap.add_argument("--text", default="pick up object",
                    help="full instruction (overridden by --object if given)")
    ap.add_argument("--proprio", action="store_true",
                    help="read real RIGHT-arm/hand proprio from the V2AP robot "
                         "(run in the robot conda env; needs dexcontrol/sharpa)")
    ap.add_argument("--proprio-steps", type=int, default=4,
                    help="number of proprio timesteps (>=4; server reads [-4],[-1])")
    ap.add_argument("--proprio-dt", type=float, default=0.1,
                    help="seconds between proprio samples")
    ap.add_argument("--no-gripper", action="store_true",
                    help="with --proprio, skip connecting the right hand (gripper=open)")
    ap.add_argument("--robot-name", default=None,
                    help="ROBOT_NAME for dexcontrol (else use env / source setup.sh)")
    ap.add_argument("--robot-ip", default=None, help="ROBOT_IP for dexcontrol")
    args = ap.parse_args()

    # GraspVLA text instruction: "pick up {object}" (--object) or a full --text
    text = f"pick up {args.object}" if args.object else args.text
    print(f'text: "{text}"')

    # read proprio first: if the robot isn't reachable we want to know before capture
    proprio = None
    if args.proprio:
        from robot_proprio import RightArmProprioReader
        with RightArmProprioReader(read_gripper=not args.no_gripper,
                                   robot_name=args.robot_name,
                                   robot_ip=args.robot_ip) as reader:
            proprio = reader.read_history(args.proprio_steps, args.proprio_dt)
        last = proprio[-1]
        print(f"proprio[-1] (GraspVLA frame): pos={np.round(last[:3],4).tolist()} "
              f"rpy={np.round(last[3:6],4).tolist()} gripper={last[6]:+.3f}  "
              f"({len(proprio)} steps)")

    print(f"capturing front={args.front}  side={args.side} ...")
    front_rgb, side_rgb = get_views(args.front, args.side)

    print("GraspVLA RGB views:")
    ok = _check("front_view_image", front_rgb) & _check("side_view_image", side_rgb)
    if not ok:
        raise SystemExit("ERROR: views do not match GraspVLA's (256,256,3) uint8 contract")

    # assemble the full obs dict (images + proprio + text), exactly what serve.py wants
    obs = build_obs(front_rgb, side_rgb, text=text, proprio=proprio)
    proprio_arr = np.asarray(obs["proprio_array"], dtype=np.float32)  # (steps, 7)

    # write all three GraspVLA inputs into one folder
    out = save_bundle(args.outdir, front_rgb, side_rgb, proprio_arr, text)
    if args.npz:
        np.savez(args.npz, front_view_image=front_rgb, side_view_image=side_rgb,
                 proprio_array=proprio_arr, text=text)
        print(f"saved arrays to {args.npz}")

    print(f"\nGraspVLA input bundle in '{out}/':")
    print(f"  front_view.png       256x256x3 uint8 RGB  (front_view_image)")
    print(f"  side_view.png        256x256x3 uint8 RGB  (side_view_image)")
    print(f"  proprio_array.npy    {proprio_arr.shape} float32  (x,y,z,r,p,y,gripper)"
          f"{'  [DUMMY]' if not args.proprio else ''}")
    print(f'  text.txt             "{text}"')
    print(f"  obs.pkl              pickled dict ready to send to the server")
    print("OK: complete GraspVLA input written.")


if __name__ == "__main__":
    main()
