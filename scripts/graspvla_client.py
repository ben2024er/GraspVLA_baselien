#!/usr/bin/env python3
"""GraspVLA client —— 雷蛇(blade15)对接 A6000 server 的闭环模块。

职责:把雷蛇打包好的 obs 发给 A6000 上的 GraspVLA server，拿回动作，
并提供 base<->graspvla 坐标换算、动作增量落到机器人 base 系的工具函数。

设计原则:server 不改(官方 serve.py)。本模块只做 client 侧。

依赖:pyzmq, numpy, transforms3d (动作姿态合成用)。

────────────────────────────────────────────────────────────────────────
通信契约(与 GraspVLA/vla_network/scripts/serve.py 对齐):
  socket: ZMQ REQ -> tcp://<A6000_IP>:6666
  发: pickle.dumps(obs)
  收: pickle.loads(bytes) -> {'info','env_id','result','debug'}

obs 格式(雷蛇 capture_views.py 已产出):
  {
    'text': 'pick up bottle',            # 真实英文类别名(见 res/category_list.txt),勿用 "object"
    'front_view_image': [HxWx3 uint8],   # 256x256(或 JPEG bytes 且 compressed=True)
    'side_view_image':  [HxWx3 uint8],
    'proprio_array': [4x (7,) float32],  # [x,y,z,roll,pitch,yaw,grip],graspvla 系,sxyz,grip∈[-1,1]
    'compressed': False,
  }
reply:
  result: (N,7) 动作序列,每步 [Δx,Δy,Δz,Δroll,Δpitch,Δyaw, grip]  ← graspvla 系增量
          (server 已做 ×2 插值;grip∈{-1,0,1}: -1关 / 1开 / 0不变)
  debug.pose: (pos3, rpy3) 抓取 goal(graspvla 系,绝对位姿)
  debug.bbox: (front_bbox, side_bbox) 224 尺度

坐标映射(已锁定): graspvla_base = Dexmate base 下移 0.75m,纯平移无旋转
  proprio 发送前: pos_z -= 0.75
  goal 转回 base : pos_z += 0.75
  动作增量(Δ):平移不影响增量 → 直接用,无需偏移
────────────────────────────────────────────────────────────────────────
"""
import pickle

import numpy as np

Z_OFFSET = 0.75  # graspvla_base = base - (0,0,Z_OFFSET)


class GraspVLAClient:
    def __init__(self, host, port=6666, timeout_s=30):
        import zmq
        self._zmq = zmq
        self.ctx = zmq.Context.instance()
        self.host, self.port, self.timeout_s = host, port, timeout_s
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, int(self.timeout_s * 1000))
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def infer(self, obs: dict) -> dict:
        """发 obs，收 reply。超时会重建 socket 并抛 TimeoutError。"""
        self.sock.send(pickle.dumps(obs))
        try:
            return pickle.loads(self.sock.recv())
        except self._zmq.Again:
            self.sock.close()
            self._connect()  # REQ 超时后状态损坏,必须重连
            raise TimeoutError(f"server {self.host}:{self.port} 未在 {self.timeout_s}s 内响应")


# ── 坐标换算工具 ───────────────────────────────────────────────────────
def base_pos_to_graspvla(pos_base):
    """Dexmate base 系 xyz -> graspvla 系 xyz(z 减 0.75)。"""
    p = np.asarray(pos_base, dtype=float).copy()
    p[2] -= Z_OFFSET
    return p


def graspvla_pos_to_base(pos_g):
    """graspvla 系 xyz -> Dexmate base 系 xyz(z 加 0.75)。用于把 goal 转回 base。"""
    p = np.asarray(pos_g, dtype=float).copy()
    p[2] += Z_OFFSET
    return p


def apply_delta_action(cur_pos_base, cur_R_base, delta):
    """把一个动作增量落到 Dexmate base 系的目标位姿。

    delta: [Δx,Δy,Δz,Δroll,Δpitch,Δyaw, grip]，graspvla 系(=base，纯平移)。
    返回 (target_pos_base(3,), target_R_base(3x3), gripper_cmd)。
    姿态合成与 Franka 控制器一致: R_target = dR @ R_current。
    gripper_cmd ∈ {-1,0,1}: -1 关 / 1 开 / 0 不变 → 映射到 Sharpa 虚拟夹爪。
    """
    import transforms3d as t3d
    delta = np.asarray(delta, dtype=float)
    target_pos = np.asarray(cur_pos_base, float) + delta[:3]  # 平移增量两系相同
    dR = t3d.euler.euler2mat(delta[3], delta[4], delta[5], axes="sxyz")
    target_R = dR @ np.asarray(cur_R_base, float)
    return target_pos, target_R, float(delta[6])


# ── 自测:对着 live server 发 brio_out/obs.pkl ──────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6666)
    ap.add_argument("--obs", default="scripts/brio_out/obs.pkl")
    ap.add_argument("--text", default=None, help="覆盖 obs 里的 text(用真实类别名)")
    args = ap.parse_args()

    with open(args.obs, "rb") as f:
        obs = pickle.load(f)
    if args.text:
        obs["text"] = args.text

    cli = GraspVLAClient(args.host, args.port)
    print(f"[client] -> tcp://{args.host}:{args.port}  text={obs['text']!r}")
    reply = cli.infer(obs)
    print(f"[client] info={reply.get('info')}")
    act = np.asarray(reply["result"])
    print(f"[client] action seq shape={act.shape}  first={np.round(act[0],4)}  last={np.round(act[-1],4)}")
    pos, ori = reply["debug"]["pose"]
    print(f"[client] goal(graspvla)={np.round(np.asarray(pos),3)}  -> base={np.round(graspvla_pos_to_base(pos),3)}")
    print("[client] ✓ 闭环 OK" )
