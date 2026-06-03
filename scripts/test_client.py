#!/usr/bin/env python3
"""GraspVLA server 本地测试 client。

读取一个 obs.pkl（雷蛇打包的模型输入），通过 ZMQ 发给 server，
打印返回的 action / goal / bbox，并把预测 bbox 画到两路视图存图。

无需机器人、无需雷蛇即可验证 server 是否正常推理。

用法：
  python scripts/test_client.py                       # 连本地 6666，用 scripts/brio_out/obs.pkl
  python scripts/test_client.py --host <A6000_IP> --port 6666 --obs scripts/brio_out/obs.pkl
"""
import argparse
import io
import pickle
import time

import numpy as np

THIS_DIR_OBS = "scripts/brio_out/obs.pkl"


def decode_image(x):
    """obs 里的图既可能是 ndarray，也可能是 JPEG bytes（compressed=True 时）。"""
    if isinstance(x, (bytes, bytearray)):
        from PIL import Image
        return np.array(Image.open(io.BytesIO(x)))
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="6666")
    ap.add_argument("--obs", default=THIS_DIR_OBS, help="obs.pkl 路径")
    ap.add_argument("--out", default="scripts/brio_out/bbox_vis.png", help="bbox 可视化输出")
    ap.add_argument("--timeout", type=int, default=60, help="recv 超时（秒）")
    args = ap.parse_args()

    import zmq

    with open(args.obs, "rb") as f:
        obs = pickle.load(f)
    print(f"[client] loaded obs from {args.obs}")
    print(f"         text          = {obs.get('text')!r}")
    print(f"         compressed    = {obs.get('compressed', False)}")
    print(f"         proprio[-1]   = {np.asarray(obs['proprio_array'][-1])}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout * 1000)
    sock.setsockopt(zmq.LINGER, 0)
    addr = f"tcp://{args.host}:{args.port}"
    sock.connect(addr)
    print(f"[client] connected {addr}, sending obs ...")

    t0 = time.time()
    sock.send(pickle.dumps(obs))
    try:
        reply = pickle.loads(sock.recv())
    except zmq.Again:
        print(f"[client] ✗ 超时 {args.timeout}s 没收到回复。确认 server 在 {addr} 运行。")
        return
    dt = time.time() - t0
    print(f"[client] ✓ got reply in {dt:.3f}s")

    print(f"         info          = {reply.get('info')}")
    action = np.asarray(reply.get("result"))
    print(f"         action seq    = shape {action.shape}  (每步 [Δx,Δy,Δz,Δr,Δp,Δy,grip])")
    if action.size:
        print(f"         action[0]     = {np.round(action[0], 4)}")
        print(f"         action[-1]    = {np.round(action[-1], 4)}")
    debug = reply.get("debug", {})
    if "pose" in debug:
        pos, ori = debug["pose"]
        pos = np.asarray(pos)
        print(f"         goal pos(graspvla) = {np.round(pos, 4)}")
        print(f"         goal pos(base)     = {np.round(pos + np.array([0, 0, 0.75]), 4)}  (+0.75 z 回 Dexmate base)")
        print(f"         goal ori           = {np.round(np.asarray(ori), 4)}")

    # 画 bbox
    if "bbox" in debug:
        try:
            import cv2
        except ImportError:
            print("[client] 无 cv2，跳过 bbox 可视化")
            return
        front = decode_image(obs["front_view_image"][0]).copy()
        side = decode_image(obs["side_view_image"][0]).copy()
        for img, bb in zip((front, side), debug["bbox"]):
            bb = (np.asarray(bb) / 224 * img.shape[0]).astype(int)
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        merged = np.concatenate([front, side], axis=1)
        cv2.imwrite(args.out, cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
        print(f"[client] bbox 可视化已存: {args.out}  （绿框应套住目标物体）")


if __name__ == "__main__":
    main()
