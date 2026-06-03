#!/usr/bin/env python3
"""Read the RIGHT-arm + RIGHT-hand proprio for GraspVLA from the V2AP robot.

Produces the 7-D proprio GraspVLA's server expects:
    [x, y, z, roll, pitch, yaw, gripper]
  - x,y,z   : RIGHT end-effector position (frame "R_ee"), via Dexmate FK,
              expressed in the GraspVLA base frame (Dexmate base z minus 0.75,
              see project README section 2: p_graspvla = p_base - (0,0,0.75)).
  - r,p,y   : RIGHT eef orientation as Euler 'sxyz' (== scipy extrinsic 'xyz').
              NOTE passthrough only -- the documented R_ee vs GraspVLA sim-EEF
              +/-90deg z offset (README section 5) is NOT applied here.
  - gripper : RIGHT hand open fraction mapped to [-1, 1]  (+1 open, -1 closed).
              The server remaps [-1,1] -> [0,1] via (g+1)/2 (serve.py).

Requires the V2AP robot stack (dexcontrol, pinocchio, pink, sharpa,
dexmate_urdf) -- i.e. run in the robot conda env (e.g. sharpa-dexmate-tmp),
NOT the GraspVLA env. Import is lazy so capture_views.py still runs camera-only
in the GraspVLA env.

This reads ONLY the right arm and right hand, as requested.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# default location of the V2AP-demo repo (sibling of this scripts/ dir)
_DEFAULT_V2AP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "V2AP-demo")

# GraspVLA base frame is Dexmate base shifted down 0.75 m (README section 2)
GRASPVLA_Z_OFFSET = 0.75


class RightArmProprioReader:
    """Connects to the Dexmate robot (+ right Sharpa hand) and reads proprio."""

    def __init__(self, v2ap_path=_DEFAULT_V2AP, read_gripper=True,
                 z_offset=GRASPVLA_Z_OFFSET, robot_name=None, robot_ip=None):
        if v2ap_path not in sys.path:
            sys.path.insert(0, v2ap_path)
        self.z_offset = float(z_offset)
        self.read_gripper = read_gripper

        # dexcontrol needs ROBOT_NAME (and optionally ROBOT_IP) set before connecting
        # (V2AP-demo/setup.sh exports these). Allow passing them in explicitly.
        if robot_name:
            os.environ["ROBOT_NAME"] = robot_name
        if robot_ip:
            os.environ["ROBOT_IP"] = robot_ip
        if not os.environ.get("ROBOT_NAME") and not os.environ.get("ROBOT_CONFIG"):
            raise RuntimeError(
                "ROBOT_NAME not set. `source V2AP-demo/setup.sh` first, or pass "
                "robot_name=... (e.g. 'dm/vgd1262ab823-1p').")

        # --- imports from the robot stack (lazy; only when proprio is used) ---
        from scipy.spatial.transform import Rotation  # 'xyz' == transforms3d 'sxyz'
        from dexcontrol.robot import Robot
        from teleop.ik_utils import PinkLocalIK
        from teleop.robot_descriptions import DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES
        self._Rotation = Rotation
        self._joint_names = DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES

        # --- connect robot ---
        print("[proprio] connecting to Dexmate robot ...")
        self.robot = Robot()

        # --- build FK solver using the robot's ACTUAL (frozen) head+torso pose ---
        # FK of R_ee runs through the torso, so bake in the live torso/head
        # configuration to get an exact eef pose (we keep torso/head frozen while
        # running GraspVLA, per README section 2).
        cfg = self.robot.get_joint_pos_dict(
            component=["head", "torso", "left_arm", "right_arm"])
        default_by_component = {
            comp: np.array([cfg[n] for n in self._joint_names[comp]], dtype=np.float64)
            for comp in ["head", "torso", "left_arm", "right_arm"]
        }
        print(f"[proprio] torso={np.round(default_by_component['torso'],3).tolist()} "
              f"head={np.round(default_by_component['head'],3).tolist()}")
        self.solver = PinkLocalIK(default_by_component)

        # --- connect right hand for gripper state (optional) ---
        self.right_hand = None
        self._hand_manager = None
        self._gripper_helpers = None
        if read_gripper:
            self._connect_right_hand()

    def _connect_right_hand(self):
        from demo.hardware import connect_right_hand_only
        from demo.hand_close import read_hand_joint_pos
        from demo.virtual_gripper import HAND_PINCH_OPEN, HAND_PINCH_CLOSED
        print("[proprio] connecting right Sharpa hand ...")
        self.right_hand, self._hand_manager = connect_right_hand_only()
        # precompute the open<->closed axis over the thumb+index pinch joints (0-8)
        open9 = np.asarray(HAND_PINCH_OPEN, dtype=np.float64)[0:9]
        closed9 = np.asarray(HAND_PINCH_CLOSED, dtype=np.float64)[0:9]
        axis = open9 - closed9
        self._gripper_helpers = (read_hand_joint_pos, closed9, axis,
                                 float(axis @ axis))

    # ------------------------------------------------------------------ reads
    def _read_right_eef_pose(self):
        """Return (pos[3], rpy[3]) of R_ee in GraspVLA base frame."""
        jd = self.robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
        left = np.array([jd[n] for n in self._joint_names["left_arm"]], dtype=np.float64)
        right = np.array([jd[n] for n in self._joint_names["right_arm"]], dtype=np.float64)
        T = self.solver.fk(
            frames=["R_ee"],
            joint_pos_by_component={"left_arm": left, "right_arm": right},
        )["R_ee"].homogeneous  # (4,4) in Dexmate base ("universe") frame
        pos = T[:3, 3].copy()
        pos[2] -= self.z_offset  # Dexmate base -> GraspVLA base
        rpy = self._Rotation.from_matrix(T[:3, :3]).as_euler("xyz")  # sxyz
        return pos, rpy

    def _read_gripper(self):
        """Right-hand open fraction -> gripper in [-1,1] (+1 open, -1 closed)."""
        if self._gripper_helpers is None:
            return 1.0  # no hand connected -> report open
        read_hand_joint_pos, closed9, axis, axis_sq = self._gripper_helpers
        q9 = read_hand_joint_pos(self.right_hand)[0:9]
        frac = float((q9 - closed9) @ axis) / axis_sq if axis_sq > 0 else 0.5
        frac = min(1.0, max(0.0, frac))  # 0=closed, 1=open
        return 2.0 * frac - 1.0          # -> [-1, 1]

    def read_step(self):
        """One 7-D proprio sample [x,y,z,roll,pitch,yaw,gripper] (GraspVLA frame)."""
        pos, rpy = self._read_right_eef_pose()
        gripper = self._read_gripper() if self.read_gripper else 1.0
        return np.array([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], gripper],
                        dtype=np.float32)

    def read_history(self, n_steps=4, dt=0.1):
        """Read n_steps proprio samples (>=4 so server can index [-4] and [-1]).

        Returns a list of 7-D float32 arrays, oldest first.
        """
        n_steps = max(4, int(n_steps))
        hist = []
        for i in range(n_steps):
            hist.append(self.read_step())
            if dt > 0 and i < n_steps - 1:
                time.sleep(dt)
        return hist

    def close(self):
        try:
            if self.right_hand is not None:
                self.right_hand.stop()
            if self._hand_manager is not None:
                self._hand_manager.disconnect_all()
        except Exception as e:
            print(f"[proprio] hand shutdown warning: {e}")
        try:
            if getattr(self, "robot", None) is not None:
                self.robot.shutdown()
        except Exception as e:
            print(f"[proprio] robot shutdown warning: {e}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def get_right_arm_proprio(n_steps=4, dt=0.1, read_gripper=True,
                          v2ap_path=_DEFAULT_V2AP, z_offset=GRASPVLA_Z_OFFSET):
    """Convenience: connect, read n_steps proprio samples, disconnect."""
    with RightArmProprioReader(v2ap_path, read_gripper, z_offset) as r:
        return r.read_history(n_steps, dt)


if __name__ == "__main__":
    # quick manual test: print one proprio history
    hist = get_right_arm_proprio(n_steps=4, dt=0.1)
    for i, s in enumerate(hist):
        print(f"step {i}: pos={np.round(s[:3],4).tolist()} "
              f"rpy={np.round(s[3:6],4).tolist()} gripper={s[6]:+.3f}")
