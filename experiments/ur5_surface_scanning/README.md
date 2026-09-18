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
  ↓
100–107 TSID模型一致性、HQP约束、FT300力控、全路径扫描与鲁棒性对比
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
cd ~/ur5_ft300_model_based_control/experiments/ur5_surface_scanning
conda activate ur5-control

env -u PYTHONPATH python 97A_ur5_unknown_surface_exploration.py
env -u PYTHONPATH python 97B_reconstruct_unknown_surface.py
env -u PYTHONPATH python 97C_check_learned_surface_reachability.py
env -u PYTHONPATH python 97D_ur5_learned_surface_5n_scan.py
```

已有 NPZ 数据时，可直接从 97C 或 97D 开始。

## TSID/HQP 控制主线

旧 80 系列只在 Pinocchio 内部积分状态；新的 100–107 已接入当前 UR5
MuJoCo 模型，MuJoCo 是真实被控对象，TSID 每个 1 ms 周期读取仿真状态并
输出关节力矩。

```bash
# 检查关节/执行器映射、TCP运动学、质量矩阵和重力项
env -u PYTHONPATH python 100_validate_ur5_tsid_model.py

# 无界面运行TSID关节姿态闭环
env -u PYTHONPATH python 101_ur5_tsid_mujoco_posture.py

# 可视化运行
env -u PYTHONPATH python 101_ur5_tsid_mujoco_posture.py --viewer

# 5D TCP主任务 + 低权重关节姿态任务
env -u PYTHONPATH python 102_ur5_tsid_mujoco_se3_hierarchy.py

# HQP内部关节位置/速度/加速度和执行器力矩硬约束
env -u PYTHONPATH python 103_ur5_tsid_hqp_safety_bounds.py

# FT300接触检测、5 N建立/保持与自动撤离
env -u PYTHONPATH python 104_ur5_tsid_ft300_5n_contact.py

# TSID/HQP完整1.54 m学习曲面5 N扫描
env -u PYTHONPATH python 105_ur5_tsid_learned_surface_5n_scan.py

# 自动生成97D与105定量对比报告
env -u PYTHONPATH python 106_compare_97d_vs_tsid_hqp.py

# 七个完整路径TSID/HQP鲁棒性场景
env -u PYTHONPATH python 107_ur5_tsid_learned_surface_robustness.py
```

当前验证结果：5 组测试姿态下的 MuJoCo/Pinocchio 模型相对误差约为
`1e-10`；101 最终关节误差小于 `1e-6 deg`，无力矩饱和或 QP 失败。
TSID 的转子惯量通过 `RobotWrapper.set_rotor_inertias()` 配置，MuJoCo
被动力在写入 `data.ctrl` 前显式扣除。

102 使用 TCP XYZ + 世界系 Roll/Pitch 五维任务，将偏航自由度留给姿态
正则化；最终位置/倾斜误差为 `0.0114 mm / 0.00053 deg`。当前 TSID 1.10
Python 绑定在加入 `priority=2` 运动任务时会发生原生崩溃，因此软任务采用
同一层级的明确权重比，`priority=0` 专门用于硬约束。

103 将关节位置、速度、加速度和执行器力矩约束放入 HQP。测试中速度和
力矩约束均实际激活；控制输出不做事后裁剪，QP 失败和约束违规均为零。
执行器约束每周期根据 MuJoCo `qfrc_passive` 平移，保证约束对应最终
`data.ctrl`，而不是只约束未补偿的 TSID 广义力。

104 使用完整状态机验证 FT300 接触闭环：零偏保持、慢速接近、接触检测、
5 N 平滑建立、恒力保持和自动撤离。接触仅由 MuJoCo 计算，未在 TSID 中
重复添加刚性接触；FT300 外力通过 `J^T F_ext` 补偿。执行器边界同时根据
外力广义力和 `qfrc_passive` 动态平移，因此约束对应最终电机指令。

当前 104 结果：5 N 稳态误差接近零，最大实测力 `5.007 N`，最大电机力矩
`22.951 Nm`，QP失败、非预期碰撞和力矩边界违规均为零，撤离后接触数为零。

105 将同一控制结构用于97B学习曲面的完整961位姿、1.538 m扫描。扫描耗时
`40.286 s`，力平均/最大误差为 `0.312/2.672 N`，TCP平均/最大误差为
`0.193/1.107 mm`；QP失败、非预期碰撞及电机边界违规均为零。

106 从97D与105结果文件自动生成CSV和Markdown报告。两种控制器使用完全
相同的路径、速度和5 N目标；105在保持相当跟踪性能的同时，将安全边界放入
HQP并取消求解后力矩裁剪。报告见 `106_tsid_vs_97d_comparison.md`。

107沿用97E的7个完整路径扰动场景。TSID/HQP通过 `6/7`：工件X/Z偏移、
1°倾斜和低摩擦场景全部通过；高摩擦 `mu=0.60` 因力平均/最大误差
`0.528/7.076 N` 超阈值而失败。所有场景的QP失败、非预期碰撞和电机边界
违规均为零。逐场景报告见
`results/tsid_robustness/107_tsid_vs_97d_robustness.md`。

## 鲁棒性实验

运行97D基准控制器的全部七个完整路径场景：

```bash
env -u PYTHONPATH python 97E_ur5_learned_surface_robustness.py
```

运行TSID/HQP控制器的相同场景：

```bash
env -u PYTHONPATH python 107_ur5_tsid_learned_surface_robustness.py
```

只运行指定场景：

```bash
env -u PYTHONPATH python 97E_ur5_learned_surface_robustness.py \
  --case offset_z_plus_1mm \
  --case tilt_y_plus_1deg
```

107同样支持重复传入 `--case` 和 `--keep-traces`。

使用 `--keep-traces` 可保存每个场景的完整时间序列。默认只保存紧凑汇总：

```text
results/robustness/97E_robustness_summary.csv
results/robustness/97E_robustness_summary.npz
results/tsid_robustness/107_tsid_robustness_summary.csv
results/tsid_robustness/107_tsid_robustness_summary.npz
results/tsid_robustness/107_tsid_vs_97d_robustness.md
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
| TSID/HQP | `100_*` 至 `107_*`、`ur5_tsid_mujoco_common.py` |

`86_build_inspection_probe_model.py` 的 UR3 基础模型来源已显式指向 `../archive/ur3_ft300_learning/`；其输出仍保存在本目录。
