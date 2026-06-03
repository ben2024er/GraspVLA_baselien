# 交接说明（HANDOFF）

写给**接手在"与机器人直连的电脑"上继续开发的人 / agent**。
先通读 `README.md`(项目权威文档),本文件只补充交接相关的实操信息。

> 交接时间:2026-06-02 ｜ 交接来源机器:开发调研机(非机器人直连)

---

## 1. 一句话现状

GraspVLA→Dexmate 部署的**调研 + 坐标系设计 + 相机摆位**已完成,**两台 BRIO 相机已物理摆好**。
下一步是在机器人电脑上:**打通 BRIO 采集 → 起 GraspVLA server → 静态 bbox 验证(不动机器人)→ 再做控制集成**。

---

## 2. 接手第一步:把代码拉到机器人电脑

四个上游仓库 + 本项目文档。在机器人电脑上:

```bash
mkdir -p ~/GraspVLA_Deploy && cd ~/GraspVLA_Deploy
# 本项目文档（README/HANDOFF/脚本）从我们的仓库拉：
#   git clone <ben2024er/GraspVLA_baselien 的地址> .
# 四个上游 repo（也可直接用 scripts/clone_upstream.sh）：
git clone https://github.com/PKU-EPIC/GraspVLA.git
git clone https://github.com/MiYanDoris/GraspVLA-playground.git           # 仿真，真机用不到，可跳过
git clone https://github.com/MiYanDoris/GraspVLA-real-world-controller.git
git clone https://github.com/jiaka1chen/V2AP-demo.git
```
(若本仓库已包含这些代码,则无需重新 clone — 见本仓库实际结构。)

---

## 3. 环境 / 依赖

- **GraspVLA server**:Python 3.9.19(conda),`GraspVLA/requirements.txt`。需要 **GPU ~9GB**(官方在 RTX L40s 上 200ms)。权重:`hf download shengliangd/GraspVLA`(或百度云,见 server README,pwd=6666)。
  启动:`python3 -u -m vla_network.scripts.serve --path <model_path> --port 6666`(可加 `--compile` 提速)。
  自检:`python3 -u -m vla_network.scripts.offline_test --port 6666`(发 mock 数据,返回 ✓ 即通)。
- **V2AP-demo**:见其 `setup.sh` / `README.md`。Dexmate 用 `dexcontrol`(Zenoh,需实验室证书),Sharpa 用 `SharpaWaveSDK`。
- **相机**:BRIO 走 UVC,用 OpenCV `cv2.VideoCapture` 即可,**不需要 pyrealsense2**。

> ⚠️ **GPU 在哪**:server 可与机器人电脑同机,也可放另一台 GPU 机(client/server 走 ZMQ TCP,设 `SERVER_IP/SERVER_PORT`)。接手时先确认 GPU 资源。

---

## 4. 接手要做的事(按优先级)

1. **BRIO 采集层**:仿照 `GraspVLA-real-world-controller/vla_client/utils/cameras.py`,把 RealSense 换成 OpenCV `VideoCapture`。
   - 裁剪:中心裁成正方形(BRIO 1080p → 1080×1080)→ `cv2.resize` 到 256×256。
   - **确认相机 FOV 预设 = 78°**(Logitech 软件里设;见 README §3.1)。
   - 写个双路预览,按 README §3.3 的肉眼准则核对 front/side 视角。
2. **起 server + offline_test** 自检通。
3. **静态感知验证(不动机器人)**:放物体 → 拍 front+side + dummy proprio → 发 server → 画回 `debug.bbox`,确认绿框套住物体;`goal` 经 `+ (0,0,0.75)` 转回 base 看是否落在物体真实位置。**这一步通过 = 相机+坐标系链路 OK。**
4. **坐标变换**:proprio(base→graspvla 仅 z−0.75,欧拉 `sxyz`)、action(透传)、Sharpa 虚拟 TCP、gripper 开/合映射。
5. **接 V2AP**:eef 增量 → `ArmIKManager`(Pink IK)→ Dexmate 臂;gripper → `virtual_gripper.py`+`hand_close.py`;过 `SmoothingAndSafetyManager`。
6. **真机抓取**,按 README §5 开放项逐个调。

---

## 5. 必须知道的坑 / 注意点

- **坐标映射只动 z**:`p_graspvla = p_base − (0,0,0.75)`。proprio 喂 server 前 z 减 0.75;server 回的 goal/action 是 graspvla 系,转回 base 要 z 加 0.75。x/y/姿态/增量都不变。
- **运行期冻结 torso + head**:否则 base↔臂、base↔相机不再刚性,模型几何全错。
- **不要**跑 controller 的 `docker calibrate_camera`,**不要**用 `res/front_ref.png|side_ref.png`(都是 Franka,对 Dexmate 不适用)。相机靠肉眼准则对齐。
- **内参/外参不用测**(README §3.2)。
- **EEF 姿态对齐是独立待解项**:Dexmate `L_ee/R_ee` 有 ±90° z 旋转 vs GraspVLA sim-EEF;平移搞定后若夹爪朝向不对,从这里查(README §5)。
- **安全**:先用 V2AP 的 `SmoothingAndSafetyManager`(碰撞检查)+ 限速;首次真机抓取手放急停旁。bring-up 先单桌、单物体、物体放工作区中心(base x 0.35~0.75)。
- **GraspVLA 是纯 RGB**,深度/点云喂不进去(架构限制)。

---

## 6. 关键文件速查

| 需求 | 文件 |
|---|---|
| server I/O / 输入格式 | `GraspVLA/vla_network/scripts/serve.py` |
| 服务自检 / 可视化 | `GraspVLA/vla_network/scripts/offline_test.py` |
| 相机采集/裁剪(改造蓝本) | `GraspVLA-real-world-controller/vla_client/utils/cameras.py` |
| Franka proprio/动作执行(参照) | `GraspVLA-real-world-controller/vla_client/controllers/franka_ros_controller.py` |
| 抓取主循环(参照) | `GraspVLA-real-world-controller/vla_client/modes/grasp_mode.py` |
| 相机摆位图(几何来源) | `GraspVLA-real-world-controller/res/camera_setup.jpg` |
| Dexmate IK / 控制 / 安全 | `V2AP-demo/teleop/arm_hand_control.py` |
| Dexmate/Sharpa 规格、frame | `V2AP-demo/teleop/robot_descriptions.py` |
| 虚拟夹爪 / stall-close | `V2AP-demo/demo/virtual_gripper.py`, `demo/hand_close.py` |
| URDF(base/arm_center/L_ee 等) | `V2AP-demo/dexmate/dexmate-urdf/robots/humanoid/vega_1/` |

---

## 7. 还没做 / 已知不确定

- 静态验证还没跑过,**相机视角是否真的匹配训练分布,尚未实证**(这是第一个要验的)。
- EEF 姿态旋转、单臂 y 居中、甜点区 x 真值都还是默认假设(README §5),等实测调。
- BRIO 78° 下方裁的有效视野与训练相机的精确匹配度,需在静态验证里看 bbox 是否准来反推。
