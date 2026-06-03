#!/usr/bin/env python3
"""GraspVLA → Dexmate 闭环编排骨架(动作侧集成起点)。

在雷蛇(blade15)上运行:采集→真实 proprio→发 A6000 server→拿动作增量→
经 sim-EEF 姿态对齐转成 R_ee 目标位姿→V2AP IK+安全层→Dexmate;grip→Sharpa。

【本文件的分工】
  ✅ 已写对、与机器人无关、最易错的部分:坐标数学
     - proprio 装配(R_ee FK → sim-EEF → graspvla 系 7 维)
     - 动作增量落位姿(Franka 控制器同款:R_target = ΔR @ R_cur)
     - sim-EEF ↔ R_ee 姿态对齐(单一固定变换 T_REE_TO_SIMEEF)
     - graspvla ↔ base 的 z±0.75
  🟡 需雷蛇填实现(已留抽象接口 + TODO,指向 V2AP):
     - DexmateRobot.get_images / get_ree_pose_base / get_gripper / move_ree / set_gripper
  ⚠️ 需标定:T_REE_TO_SIMEEF(含 ±90° z + Sharpa 虚拟 TCP 偏移),见下方常量。

依赖:numpy, scipy(雷蛇 env 已有);GraspVLAClient(scripts/graspvla_client.py)。
干跑(不接机器人,验编排逻辑):python scripts/run_graspvla_dexmate.py --dry-run
"""
import argparse
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from graspvla_client import GraspVLAClient, Z_OFFSET  # 同目录

# ─────────────────────────────────────────────────────────────────────────
# 【需标定】sim-EEF 相对 R_ee 的固定变换(4x4,在 R_ee 局部系)。
#   GraspVLA 的 proprio/动作都基于 "sim-EEF" 位姿;Dexmate 的臂末端是 R_ee。
#   这个变换吸收两件事:
#     (1) R_ee 相对夹爪/sim-EEF 的姿态差(Dexmate L_ee/R_ee 有 ±90° z;见 README §5)
#     (2) Sharpa 虚拟 TCP 的平移偏移(两指指尖中点,对应 Franka sim-EEF 点)
#   先给一个待验证先验:绕 z 转 -90° + 沿 z 一个 TCP 偏移。务必用"小幅试动作 + 看夹爪朝向"标定。
EEF_ALIGN_RPY_SXYZ = (0.0, 0.0, -np.pi / 2)   # TODO 标定:±90° z(方向待定)
EEF_ALIGN_XYZ      = (0.0, 0.0, 0.0)           # TODO 标定:Sharpa 虚拟 TCP 偏移(m)

def _make_T(rpy_sxyz, xyz):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("xyz", rpy_sxyz).as_matrix()  # scipy 'xyz'=extrinsic≡t3d 'sxyz'
    T[:3, 3] = xyz
    return T

T_REE_TO_SIMEEF = _make_T(EEF_ALIGN_RPY_SXYZ, EEF_ALIGN_XYZ)
T_SIMEEF_TO_REE = np.linalg.inv(T_REE_TO_SIMEEF)
GRASPVLA_SHIFT = np.array([0.0, 0.0, Z_OFFSET])  # base = graspvla + shift

# ─────────────────────────────────────────────────────────────────────────
# 坐标数学(已验证正确,无需机器人)
def ree_pose_to_proprio(T_base_ree: np.ndarray, gripper_m11: float) -> np.ndarray:
    """R_ee 在 base 系的 4x4 位姿 + 夹爪状态 → GraspVLA 7 维 proprio。
    gripper_m11 ∈ [-1,1](1 开 / -1 关),server 端会再 (x+1)/2 → [0,1]。"""
    T_base_simeef = T_base_ree @ T_REE_TO_SIMEEF
    pos_base = T_base_simeef[:3, 3]
    pos_g = pos_base - GRASPVLA_SHIFT                       # base → graspvla(仅 z-0.75)
    rpy = Rot.from_matrix(T_base_simeef[:3, :3]).as_euler("xyz")  # 姿态两系相同
    return np.array([*pos_g, *rpy, gripper_m11], dtype=np.float32)


def delta_to_ree_target(T_base_ree_cur: np.ndarray, delta: np.ndarray):
    """当前 R_ee 位姿(base 4x4) + 一个动作增量 → 下一 R_ee 目标位姿(base 4x4) + 夹爪指令。
    delta=[Δx,Δy,Δz,Δr,Δp,Δy,grip] 在 graspvla/sim-EEF 系(纯平移→增量两系相同)。
    姿态合成与 Franka 控制器一致:R_target = ΔR @ R_cur。"""
    T_base_simeef_cur = T_base_ree_cur @ T_REE_TO_SIMEEF
    pos_cur = T_base_simeef_cur[:3, 3]
    R_cur = T_base_simeef_cur[:3, :3]
    pos_tgt = pos_cur + np.asarray(delta[:3], float)       # 平移增量两系相同
    dR = Rot.from_euler("xyz", np.asarray(delta[3:6], float)).as_matrix()
    R_tgt = dR @ R_cur
    T_base_simeef_tgt = np.eye(4); T_base_simeef_tgt[:3, :3] = R_tgt; T_base_simeef_tgt[:3, 3] = pos_tgt
    T_base_ree_tgt = T_base_simeef_tgt @ T_SIMEEF_TO_REE    # 转回 R_ee 给 IK
    return T_base_ree_tgt, float(delta[6])

# ─────────────────────────────────────────────────────────────────────────
# 【需雷蛇填实现】机器人接口 —— 用 V2AP 接上 Dexmate + Sharpa
class DexmateRobot:
    """把下面每个方法接到 V2AP 的真实实现(见各 TODO 指向的文件)。"""

    def get_images(self):
        """→ (front_rgb 256x256x3 uint8, side_rgb 256x256x3 uint8)。
        TODO: scripts/capture_views.py / brio_capture.py 的两路 BRIO 采集+裁剪。"""
        raise NotImplementedError

    def get_ree_pose_base(self) -> np.ndarray:
        """→ R_ee 在 base 系的 4x4 位姿(由当前关节 FK)。
        TODO: V2AP ik_utils.py 的 fk(['R_ee'], joint_pos);冻结 torso/head。"""
        raise NotImplementedError

    def get_gripper_m11(self) -> float:
        """→ 当前夹爪状态映射到 [-1,1](1 开 / -1 关)。
        TODO: 由 Sharpa 手当前开合或上一次指令推断。"""
        raise NotImplementedError

    def move_ree(self, T_base_ree_target: np.ndarray):
        """把 R_ee 移到目标 4x4 位姿(base 系)。
        TODO: V2AP ArmIKManager.get_arm_action({'R_ee': T}) → SmoothingAndSafetyManager
              → robot.set_joint_pos。务必走碰撞检查+限速。"""
        raise NotImplementedError

    def set_gripper(self, cmd: float):
        """cmd ∈ {-1,0,1}:-1 关 / 1 开 / 0 不变 → Sharpa 虚拟夹爪。
        TODO: demo/virtual_gripper.py(拇指+食指)+ demo/hand_close.py(stall-close)。"""
        raise NotImplementedError


class DryRunRobot(DexmateRobot):
    """干跑用:用已采 obs 的图 + 一个固定 R_ee 位姿,验证编排/坐标逻辑(不接机器人)。"""
    def __init__(self, obs_path="scripts/brio_out/obs.pkl"):
        import pickle
        self.obs = pickle.load(open(obs_path, "rb"))
        # 一个示意 R_ee 位姿:base 系,工作区上方,单位姿态
        self._T = np.eye(4); self._T[:3, 3] = [0.5, 0.0, 0.95]
        self._grip = 1.0
    def get_images(self):
        return np.asarray(self.obs["front_view_image"][0]), np.asarray(self.obs["side_view_image"][0])
    def get_ree_pose_base(self): return self._T.copy()
    def get_gripper_m11(self): return self._grip
    def move_ree(self, T): self._T = T.copy()
    def set_gripper(self, cmd):
        if cmd in (-1.0, 1.0): self._grip = cmd

# ─────────────────────────────────────────────────────────────────────────
def run(robot: DexmateRobot, host, port, text, max_steps=50, hz=3.0):
    cli = GraspVLAClient(host, port)
    hist = deque(maxlen=4)
    print(f"[run] text={text!r}  server={host}:{port}")
    for step in range(max_steps):
        front, side = robot.get_images()
        T_ree = robot.get_ree_pose_base()
        proprio = ree_pose_to_proprio(T_ree, robot.get_gripper_m11())
        hist.append(proprio)
        while len(hist) < 4:
            hist.appendleft(proprio)
        obs = {
            "text": text,
            "front_view_image": [front], "side_view_image": [side],
            "proprio_array": [np.asarray(p) for p in hist],
            "compressed": False,
        }
        reply = cli.infer(obs)
        actions = np.asarray(reply["result"])
        goal_g = np.asarray(reply["debug"]["pose"][0])
        print(f"[step {step}] proprio_simeef(graspvla)={np.round(proprio[:3],3)} "
              f"goal(base)={np.round(goal_g + GRASPVLA_SHIFT,3)} actions={actions.shape}")
        # 逐步执行该 chunk(已 ×2 插值)。简单策略:执行整段后再重采(闭环)。
        for delta in actions:
            T_tgt, grip = delta_to_ree_target(robot.get_ree_pose_base(), delta)
            robot.move_ree(T_tgt)
            if grip in (-1.0, 1.0):
                robot.set_gripper(grip)
        time.sleep(1.0 / hz)
        # TODO 终止条件:夹爪闭合且抬起完成 / 多步无位移 / 人工停止。
    print("[run] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="128.32.164.89")
    ap.add_argument("--port", type=int, default=6666)
    ap.add_argument("--text", default="pick up bottle")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="不接机器人,用已采 obs 验证编排+坐标逻辑")
    args = ap.parse_args()
    robot = DryRunRobot() if args.dry_run else DexmateRobot()  # 真机:换成你的 V2AP 实现子类
    run(robot, args.host, args.port, args.text, args.max_steps)


if __name__ == "__main__":
    main()
