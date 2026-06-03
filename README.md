# GraspVLA → Dexmate 部署项目

把 **GraspVLA**(抓取基础模型,原为 Franka + 平行夹爪)部署到 **Dexmate Vega(双臂)+ Sharpa HA4 灵巧手**。
本文件是项目的**唯一权威文档(living doc)**:记录我们在做什么、调研结论、关键参数、注意点、进度与待办。新接手的人/agent 请先通读本文件 + `HANDOFF.md`。

> 最近更新:2026-06-02

---

## ⭐ 当前进度(Status）

| 阶段 | 状态 |
|---|---|
| 调研 GraspVLA 输入输出 / 架构 | ✅ 完成 |
| 坐标系映射设计(graspvla_base) | ✅ 锁定 `p_graspvla = p_base − (0,0,0.75)` |
| 相机选型 + 摆位几何 | ✅ Logitech BRIO ×2,坐标已换算到 Dexmate base |
| **物理摆放 front + side 相机** | ✅ **已摆放完成**（2026-06-02） |
| BRIO 采集层(OpenCV)+ 78° 裁剪 → 256×256 | ⏳ 待做（next） |
| 起 GraspVLA server(需 GPU,~9GB) | ⏳ 待做 |
| 静态感知验证(bbox/goal,不动机器人) | ⏳ 待做 |
| 坐标变换代码 + 接 V2AP IK/安全层 | ⏳ 待做 |
| 真机抓取实测 | ⏳ 待做 |

**下一步(立即)**:打通 BRIO 采集 + 肉眼核对相机视角 → 起 server → 静态 bbox 感知验证(全程不动机器人,先把"相机+坐标系"链路验证掉)。详见 §7。

**开发机迁移**:接下来开发将搬到**与机器人直连的电脑**上进行。交接说明见 `HANDOFF.md`。

---

## 0. 仓库说明(同目录下)

| 目录 | 来源 | 作用 |
|---|---|---|
| `GraspVLA/` | github.com/PKU-EPIC/GraspVLA | 模型 server（`vla_network/`，权威 I/O / 架构）。权重在 HuggingFace `shengliangd/GraspVLA`（~9GB），需单独下载 |
| `GraspVLA-playground/` | github.com/MiYanDoris/GraspVLA-playground | 仿真评测（robosuite/LIBERO，Panda）。**784M，仅参考/不用于真机** |
| `GraspVLA-real-world-controller/` | github.com/MiYanDoris/GraspVLA-real-world-controller | Franka 实机客户端（数据流/相机标定/控制模板）。**改造蓝本** |
| `V2AP-demo/` | github.com/jiaka1chen/V2AP-demo | 我们的 Dexmate + Sharpa 硬件栈（IK / 虚拟夹爪 / 安全层）。**集成目标** |

> 这 4 个都是各自独立的 git 仓库（带 `.git` 和各自 remote）。

---

## 1. GraspVLA 输入输出(来自 server 源码,权威)

- **架构**:DINO+SigLIP 2D backbone → InternLM2 VLM(自回归出 bbox + 抓取 goal)→ flow-matching action expert。**纯 2D RGB,不吃深度/点云**(喂点云需重训,行不通)。
- **输入**:
  - 2× RGB(**front + side,第三人称**),各 256×256(原 640×480 中心裁剪→缩放)。
  - `proprio_array`:末端 eef 位姿,**来自机器人 FK,不来自视觉**。7 维 `[x,y,z,roll,pitch,yaw,gripper]`,欧拉角 `sxyz`,gripper 归一到 [0,1]。取 2 个时间步(`serve.py` 用 `[-4]` 和 `[-1]`)。
  - `text`:`"pick up {object}"`。
- **输出**:8 维动作序列 `[Δx,Δy,Δz,Δroll,Δpitch,Δyaw, gripper∈{-1,0,1}]`,**笛卡尔 eef 增量 + 二值夹爪**,在 base 系。
- **要点**:模型把"物体感知"(RGB→bbox+goal)和"本体自定位"(proprio→FK)**解耦** → **不依赖在 RGB 里看见自己的夹爪**,所以 Sharpa 手外观差异影响有限。
- 通信:ZMQ(client 发 obs → server 回 action)。

---

## 2. 坐标系映射(已锁定)

**GraspVLA base 系** = Franka `panda_link0`(z 上、x 前、y 左;桌面 z≈0.10,工作区中心 x=0.5,y=0)。
**Dexmate base 系** = URDF root link `base`(底盘根,在地面,z 上、x 前、y 左;桌面 z=0.85 **实测确认**)。

```
graspvla_base = Dexmate base 整体下移 0.75 m，无旋转
p_graspvla = p_base − (0, 0, 0.75)
反向：       p_base = p_graspvla + (0, 0, 0.75)
```
- x_offset = 0(甜点区中心设在 base x=0.5)
- y_offset = 0
- z_offset = 0.85 − 0.10 = **0.75**
- 结果:**只有位置 z 减 0.75;x/y、姿态、action 增量全透传**。`proprio_z = base_TCP_z − 0.75`。

**为什么是 0.75 而不是 0.85**:GraspVLA 的桌面不在 z=0,而在 z≈0.10(side 相机 look-at=(0.5,0,0.1) 即桌面中心)。所以 `z_offset = Dexmate桌面(0.85) − GraspVLA桌面(0.10) = 0.75`。

**硬约束**:运行期间**冻结 torso(3-DOF)+ head**,保证 base↔臂、base↔相机刚性(等价 Franka 整机机架)。V2AP 默认 `TORSO_DEFAULT_JOINT_POS=[1.2,2.27,0.5]`、`HEAD_DEFAULT`,IK 中本就锁 torso/head。

---

## 3. 相机

### 3.1 硬件 & 设置
- 型号:**Logitech BRIO**(UVC,VendorID 1133 / ProductID 0x085e)。纯 RGB,GraspVLA 只要 RGB,正好够用。
- 需要 **2 台**(front + side,第三人称)。

> ⚠️ **设置提醒:把 BRIO 的 FOV 预设设为 78°**(≈ RealSense D435/D415 的 69°H×42°V,与训练相机最接近)。配置相机软件时**务必检查这一项**。
> - 65° → V≈35°,太窄,**不要用**(README 要求 H、V 均 >43°)。
> - 78° → H≈70° / V≈43°,**首选**。
> - 90° → H≈82° / V≈52°,偏广,需要改裁剪。
> 这是**设置项,不是测量**。

### 3.2 内参/外参:**不需要测**
- controller 读了内参 `k_real` 但**只 print、从不使用**(`cameras.py:14-15`);裁剪是**写死像素**(640×480→480×480→256×256,`cameras.py:22-31`)。
- 外参**从不计算/存储**;原版"标定"只是把预渲染参考图叠加、手动挪相机对齐。
- proprio/action 走机器人 FK/TF,**与相机无关**。
- → **结论:跑 GraspVLA 不需要相机内参、外参数值。** 只需把相机物理摆对 + FOV 设 78°。

### 3.3 相机摆位(来自 `GraspVLA-real-world-controller/res/camera_setup.jpg`,已换算到 Dexmate base)

| 项 | GraspVLA base(原图) | **Dexmate base(实摆用)** |
|---|---|---|
| 工作区 x | 0.35 ~ 0.75 m | **0.35 ~ 0.75 m** |
| 工作区 y | −0.25 ~ 0.25 m | **−0.25 ~ 0.25 m** |
| 工作区 z | −0.1 ~ 0.2 m | **0.65 ~ 0.95 m**(桌面 0.85 在其中) |
| **Front 相机位置** | (1.35, 0.0, 0.53) | **(1.35, 0.0, 1.28)** = 离地 1.28m = 桌面上方 0.43m |
| Front look-at | (0.2, 0, 0) | **(0.2, 0, 0.75)** |
| **Side 相机位置** | (0.5, 0.69, 0.5) | **(0.5, 0.69, 1.25)** = 离地 1.25m = 桌面上方 0.40m |
| Side look-at | (0.5, 0, 0.1) | **(0.5, 0, 0.85)** |

> 注意:图里的 z 是 **base 系下离基座原点**的高度,不是"离桌面"。GraspVLA 系 z=0 在 panda_link0 原点、桌面在 z=0.10;Dexmate 系 z=0 在地面、桌面在 z=0.85。

摆完用**肉眼准则**核对(**不要**用 Franka 的 `front_ref/side_ref` 绿图、**不要**跑 `docker calibrate_camera`):
- Front:机器人/工作台大致居中,桌面水平。
- Side:画面中心十字对到工作区中心,桌沿水平。

**待核实**:(1) Dexmate base 的 +x 确为前向、+y 为左(V2AP 把桌子放在 x=1.1>0,基本可确认);(2) side 相机放 +y 还是 −y 侧,按现场空间和所用手臂定,必要时 y 取负。

---

## 4. 动作侧迁移(V2AP 已有的现成件)

GraspVLA 的动作接口是笛卡尔 eef + 二值夹爪,**embodiment-agnostic**,V2AP 已备齐所有积木:

| GraspVLA 需要 | V2AP 现成件 |
|---|---|
| eef 增量 → 关节角 | `ArmIKManager`（Pink IK），`teleop/arm_hand_control.py` |
| gripper 开/合 | `demo/virtual_gripper.py`（拇指+食指当两指夹爪）+ `demo/hand_close.py`（stall-close） |
| eef 位姿 proprio | URDF + FK（给 Sharpa 定一个虚拟 TCP 点） |
| 平滑 + 碰撞安全 | `SmoothingAndSafetyManager`（Ruckig + pinocchio），300 Hz |

---

## 5. 开放项 / 待办(出问题时按此排查)

- [ ] **甜点区 x 真值**:若抓取前后方向系统性偏 → 调 x_offset(现设 0)。用 V2AP 已调好的抓取位姿跑 FK 取 R_ee 的 base 坐标可校准。
- [ ] **单臂 y 居中**:单臂时臂的横向可达中心未必在 base y=0 → 调 y_offset(现设 0)。
- [x] **桌面高度**:0.85 实测确认 → z_offset=0.75 锁定。
- [ ] **EEF/TCP 姿态对齐**(与平移无关,需单独标一个旋转):Dexmate `L_ee/R_ee` 相对法兰有 ±90° z 旋转(`L_ee_j0 rpy="0 0 -1.57"`、`R_ee_j0 rpy="0 0 1.57"`);GraspVLA sim-EEF = `panda_EE × REAL_EEF_TO_SIM_EEF`。
- [ ] **相机采集层改写**:controller 用 `pyrealsense2`,换 BRIO 要改成 UVC/OpenCV `VideoCapture`;并按 78° + 16:9 调整中心裁剪(`cameras.py`:中心裁正方形 H×H → resize 256)。
- [ ] **Sharpa 虚拟 TCP 定义**:对应 Franka sim-EEF 点(两指指尖中点)。
- [x] **第二张桌**:两桌等高拼成平面,先保留;远端保持空净、物体只放工作区(base x 0.35~0.75)。bring-up 阶段建议先单桌、单物体调通。

---

## 6. 部署主路线(建议顺序)

1. ✅ 摆相机(§3.3)+ 设 BRIO 78°(§3.1)。
2. ⏳ BRIO 采集层 + 78° 裁剪 → 256×256,肉眼核对视角。
3. ⏳ 冻结 torso/head(§2)。
4. ⏳ 起 GraspVLA server(HF 权重),先用 `offline_test` 发 mock 验证服务通。
5. ⏳ 静态感知验证:放物体 → 拍 front+side + dummy proprio → server 回 bbox/goal,验证相机+坐标系。
6. ⏳ 写坐标变换(proprio: 仅 z−0.75;action: 透传;Sharpa 虚拟 TCP + gripper 映射)。
7. ⏳ 接 V2AP 的 IK + 安全层 → 真机抓取,按 §5 逐项调。

---

## 7. 下一步详细说明(BRIO 采集 + 静态验证)

**为什么先做这个**:相机摆位是迁移里最容易出错、模型最敏感的一环(OOD 主来源),而验证它**完全不需要动机器人**,风险最低、信息量最大。

- **采集**:controller 用 `pyrealsense2`,BRIO 换成 OpenCV `VideoCapture`。裁剪照搬 `cameras.py` 思路:**中心裁成正方形(1080×1080 等)→ resize 256×256**。78° 预设下方裁保留约 43° 垂直视野,与 RealSense 方裁后 ~42° 一致,物体尺度能对上。
- **静态 bbox 验证(关键里程碑,不动机器人)**:放物体 → 拍 front+side + 一个 dummy proprio 发给 server → server 回 `debug.bbox + goal`。
  - 两视角绿框都套住物体 → 相机视角对 ✓
  - goal 经 `p_base = p_graspvla + (0,0,0.75)` 转回 base,看是否落在物体真实位置 → 几何对 ✓

---

## 8. 项目记忆(Claude memory 摘录)

> 以下内容同步自 Claude 持久记忆 `graspvla-dexmate-frame-mapping`,供无 Claude 环境时查阅。

- **目标**:GraspVLA(Franka+平行夹爪)部署到 Dexmate Vega + Sharpa HA4。4 个 repo 见 §0。
- **GraspVLA base** = `panda_link0`(z上/x前/y左,桌面 z≈0.1,工作区中心 x=0.5,y=0)。RGB→物体 bbox+抓取 goal(base 系);proprio(7维 eef,来自 FK,**非视觉**)只条件化 flow-matching action expert。动作=8维笛卡尔 eef 增量+二值夹爪。抓取位姿预测与相机外参绑定 → 相机摆位关键。
- **Dexmate base** = URDF root `base`(底盘,地面,z上/x前/y左)。臂挂在 3-DOF torso → arm_center → L_ee/R_ee;头部相机也在 torso 上。桌面高 0.85m。两张物理桌拼成一个平面。V2AP 全栈以 `base` 为根。
- **决策(2026-06-02)**:graspvla_base = base 下移 0.75m,无旋转。甜点区中心设在 base x=0.5。`p_graspvla = p_base − (0,0,0.75)`。仅位置 z 减 0.75;x/y、姿态、action 增量透传。
- **开放项**:见 §5(甜点区 x、单臂 y、EEF 姿态旋转)。桌面 0.85 已实测确认。
- **硬约束**:运行期冻结 torso+head 保刚性。两台第三人称相机按 §3.3 摆。Sharpa 虚拟 TCP + gripper 映射用 V2AP 现成件。
- **相机**:BRIO(UVC,纯 RGB),**FOV 设 78°**。内参/外参**不需要测**(controller 只 print 内参、裁剪写死像素、不存外参)。**不要**用 `docker calibrate_camera` 和 Franka 的 `front_ref/side_ref` 绿图,改用肉眼准则对齐。
- **GraspVLA 纯 2D RGB**(DINO+SigLIP→InternLM2+flow-matching),**不是点云**;喂点云需重训。本体差异影响小,因为模型靠 proprio(FK)自定位,不靠看见夹爪。

---

## 9. 交接

开发将迁移到与机器人直连的电脑。**新接手者请读 `HANDOFF.md`**(环境、依赖、如何跑、坑、联系方式)。
