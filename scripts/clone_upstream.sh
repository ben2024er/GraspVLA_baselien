#!/usr/bin/env bash
# 克隆 GraspVLA→Dexmate 部署所需的 4 个上游仓库。
# 在机器人电脑上、本项目根目录运行：bash scripts/clone_upstream.sh
set -euo pipefail
cd "$(dirname "$0")/.."

clone() { [ -d "$2/.git" ] && echo "已存在: $2 (skip)" || git clone "$1" "$2"; }

clone https://github.com/PKU-EPIC/GraspVLA.git                          GraspVLA
clone https://github.com/MiYanDoris/GraspVLA-real-world-controller.git  GraspVLA-real-world-controller
clone https://github.com/jiaka1chen/V2AP-demo.git                       V2AP-demo
# 仿真，真机用不到，需要再放开：
# clone https://github.com/MiYanDoris/GraspVLA-playground.git           GraspVLA-playground

echo "完成。GraspVLA 模型权重另需： hf download shengliangd/GraspVLA"
