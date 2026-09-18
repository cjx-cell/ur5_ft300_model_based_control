# UR5 + FT300 学习曲面恒力扫描

本目录是当前项目的完整 UR5 曲面扫描主线。目标是在控制器不知道解析曲面方程的条件下，先用低力扫描学习表面，再沿学习曲面完成约 1.54 m、5 N 恒力扫描。

## 主流程

```text
89 生成200 × 120 mm光栅路径
  ↓
93 验证UR5全路径可达性
  ↓
97A 低力探索并记录实际TCP/FT300数据
  ↓
97B 重建曲面位置、切向和法向
  ↓
97C 验证学习曲面的961个完整6D位姿
  ↓
97D 5 N学习曲面扫描
  ↓
97E 工件偏移、倾斜和摩擦鲁棒性验证
```

解析椭球只存在于 MuJoCo 环境中。97B 之后的控制器不调用解析曲面公式生成期望位姿。

## 97D 控制结构

- 97B 位置和法向：名义前馈轨迹。
- 法向力误差：调节沿学习法向的压入位移。
- FT300 TCP 残余 X/Y 力矩：调节工具 Roll/Pitch。
- 工具 Z/Yaw：保持学习路径的切向方向，避免摩擦力矩导致自转。
- Pinocchio：计算运动学、雅可比、RNEA 和前馈逆动力学。
- MuJoCo：提供实际接触、FT300 传感器和执行器动力学。

FT300 力矩先从传感器原点换算到探针 TCP：

\[
M_{TCP}=M_{FT300}-r_{FT300\rightarrow TCP}\times F.
\]

这样不会把探针力臂与切向摩擦产生的搬运力矩误判为姿态误差。

## 推荐运行顺序

```bash
cd ~/ur3_ft300_model_based_control/experiments/ur5_surface_scanning
conda activate robot310

env -u PYTHONPATH python 97A_ur5_unknown_surface_exploration.py
env -u PYTHONPATH python 97B_reconstruct_unknown_surface.py
env -u PYTHONPATH python 97C_check_learned_surface_reachability.py
env -u PYTHONPATH python 97D_ur5_learned_surface_5n_scan.py
```

已有 NPZ 数据时，可直接从 97C 或 97D 开始。

## 鲁棒性实验

运行全部七个完整路径场景：

```bash
env -u PYTHONPATH python 97E_ur5_learned_surface_robustness.py
```

只运行指定场景：

```bash
env -u PYTHONPATH python 97E_ur5_learned_surface_robustness.py \
  --case offset_z_plus_1mm \
  --case tilt_y_plus_1deg
```

使用 `--keep-traces` 可保存每个场景的完整时间序列。默认只保存紧凑汇总：

```text
results/robustness/97E_robustness_summary.csv
results/robustness/97E_robustness_summary.npz
```

## 演示媒体

先运行 97D 生成含 `qpos` 的结果文件，再执行：

```bash
MUJOCO_GL=egl env -u PYTHONPATH python 98_generate_ur5_scan_demo.py
```

输出位于仓库的 `docs/media/`：

- `ur5_learned_surface_5n_scan.mp4`
- `ur5_learned_surface_5n_scan.gif`

## 文件分组

| 阶段 | 文件 |
|---|---|
| 模型与工位 | `86_*` 至 `92_*`、`ur5_ft300_inspection*` |
| 路径与可达性 | `89_*`、`90A*`、`93_*` |
| 基准控制 | `94_*`、`95_*`、`96_*` |
| 未知曲面学习 | `97A_*`、`97B_*`、`97C_*` |
| 恒力扫描 | `97D_*` |
| 鲁棒性 | `97E_*`、`results/robustness/` |
| 媒体生成 | `98_*` |

`86_build_inspection_probe_model.py` 的 UR3 基础模型来源已显式指向 `../archive/ur3_ft300_learning/`；其输出仍保存在本目录。
