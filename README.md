# UR5 + FT300 学习曲面力控制与 TSID/HQP 约束控制

本项目使用 **MuJoCo + Pinocchio + TSID/HQP** 实现机械臂动力学控制、FT300 六维力传感处理、未知曲面学习、5 N 恒力扫描、关节/电机硬约束和鲁棒性验证。

当前推荐主线是 UR5 对未知曲面的两遍扫描：第一遍低力探索并学习表面，第二遍只使用学习结果完成约1.54 m恒力扫描。项目同时提供97D计算力矩基准控制器和105 TSID/HQP约束控制器，可在完全相同的路径与扰动场景下定量比较。

![UR5学习曲面5N扫描](docs/media/ur5_learned_surface_5n_scan.gif)

[下载 MP4 演示](docs/media/ur5_learned_surface_5n_scan.mp4)

## 项目结构

```text
ur5_ft300_model_based_control/
├── README.md
├── environment.yml
├── docs/
│   └── media/
│       ├── ur5_learned_surface_5n_scan.gif
│       └── ur5_learned_surface_5n_scan.mp4
├── examples/                         # 几何、运动学、动力学基础示例
└── experiments/
    ├── ur5_surface_scanning/         # 当前UR5曲面扫描主线
    │   ├── 89_* ... 107_*
    │   ├── ur5_ft300_inspection*
    │   └── results/
    │       ├── robustness/           # 97D基准鲁棒性结果
    │       └── tsid_robustness/      # TSID/HQP鲁棒性与对比报告
    └── archive/
        └── ur3_ft300_learning/       # 00–83、74/75及UR3历史模型
```

- [UR5 曲面扫描使用说明](experiments/ur5_surface_scanning/README.md)
- [UR3 早期学习实验归档说明](experiments/archive/ur3_ft300_learning/README.md)

## UR5 未知曲面主流程

```text
已知扫描区域的XY光栅布局
          ↓
97A：低力探索，采集实际TCP与FT300数据
          ↓
97B：拟合实际位置并估计表面法向
          ↓
97C：验证961个学习位姿全部可达、连续且无碰撞
          ↓
97D：仅使用97B学习曲面进行1.538 m、5 N扫描
          ↓
97E：工件偏移、倾斜和摩擦鲁棒性验证
          ↓
100–104：模型一致性、TSID闭环、笛卡尔任务、HQP约束、FT300接触
          ↓
105：TSID/HQP完整学习曲面5 N扫描
          ↓
106–107：基准对比与7场景鲁棒性对比
```

97B 之后不使用解析真实曲面生成期望位置或法向。MuJoCo 中的解析椭球只是隐藏环境与碰撞几何。

## 97D 控制器

97D 不是简单关节轨迹回放，而是“学习模型前馈 + 在线力/力矩反馈”：

| 信号 | 控制作用 |
|---|---|
| 97B 学习位置 | 名义 TCP 位置 |
| 97B 学习法向/姿态 | 名义工具姿态 |
| FT300 法向力误差 | 法向压入位移导纳 |
| FT300 TCP 残余 X/Y 力矩 | Roll/Pitch 姿态导纳 |
| FT300 完整六维 wrench | 外力广义力补偿 |
| Pinocchio RNEA | 逆动力学前馈 |

法向导纳为：

\[
M_f\ddot d_n+D_f\dot d_n=F_d-F_n.
\]

姿态导纳为：

\[
I_r\ddot\theta+D_r\dot\theta+K_r\theta=-M_{TCP,xy}.
\]

FT300 力矩在进入姿态控制前先换算到探针 TCP：

\[
M_{TCP}=M_{FT300}-r_{FT300\rightarrow TCP}\times F.
\]

这可以排除探针力臂和扫描摩擦产生的搬运力矩。绕工具 Z 轴的力矩不用于调姿，以免摩擦驱动工具偏航。

## 已验证结果

### 97C 学习曲面全路径可达性

| 指标 | 结果 |
|---|---:|
| IK 成功 | 961 / 961 |
| 最大位置残差 | 0.0142 mm |
| 最大姿态残差 | 0.0041° |
| 最小雅可比奇异值 | 0.139974 |
| 最大条件数 | 13.41 |
| 关节限位违规 | 0 |
| 非预期碰撞 | 0 |

### 97D 5 N 完整扫描

| 指标 | 结果 |
|---|---:|
| 路径长度 | 1538.2 mm |
| 实际扫描时间 | 40.286 s |
| 法向力平均 / 最大误差 | 0.322 / 2.940 N |
| TCP 平均 / 最大误差 | 0.191 / 1.081 mm |
| 修正姿态最大跟踪误差 | 0.359° |
| 最大执行器力矩 | 39.867 Nm |
| 力矩饱和 | 0% |
| 非预期碰撞 | 0 |
| 结果 | PASS |

## TSID/HQP 约束控制主线

100–107不是脱离MuJoCo的TSID示例，而是一套实际闭环控制链：MuJoCo作为被控对象和唯一接触求解器，TSID每1 ms读取仿真状态、求解任务空间逆动力学并输出关节力矩。

```text
MuJoCo q、dq、FT300 wrench
              ↓
FT300零偏/负载重力补偿与TCP力矩换算
              ↓
法向力导纳 + Roll/Pitch力矩导纳
              ↓
TSID笛卡尔任务 + HQP关节/电机硬约束
              ↓
外力 JᵀF 补偿 + MuJoCo被动力补偿
              ↓
         mj_data.ctrl
```

| 阶段 | 任务 |
|---|---|
| 100 | 验证MuJoCo/Pinocchio/TSID关节映射、TCP、质量矩阵与重力项 |
| 101 | TSID力矩实际驱动MuJoCo，不使用Pinocchio内部理想积分 |
| 102 | 五维TCP主任务与低权重关节姿态任务 |
| 103 | HQP内部关节位置、速度、加速度和最终电机指令约束 |
| 104 | FT300零偏、接近、接触检测、5 N建立/保持与撤离状态机 |
| 105 | 961位姿、1.538 m学习曲面TSID/HQP完整扫描 |
| 106 | 97D与105自动定量对比 |
| 107 | 工件偏移、倾斜和摩擦变化的7场景鲁棒性对比 |

### 105 TSID/HQP 完整扫描

| 指标 | 结果 |
|---|---:|
| 路径长度 / 位姿数 | 1538.2 mm / 961 |
| 扫描时间 | 40.286 s |
| 法向力平均 / 最大误差 | 0.312 / 2.672 N |
| TCP平均 / 最大误差 | 0.193 / 1.107 mm |
| 姿态平均 / 最大误差 | 0.069° / 0.361° |
| 最大执行器力矩 | 39.868 Nm |
| 电机边界违规 / QP失败 / 非预期碰撞 | 0 / 0 / 0 |
| QP P99求解时间 | 约3.1 μs |
| 结果 | PASS |

### 97D 与 105 对比

两套控制器使用完全相同的97B学习曲面、路径速度、5 N目标和MuJoCo接触模型。

| 指标 | 97D计算力矩 | 105 TSID/HQP |
|---|---:|---:|
| 力平均误差 | 0.322 N | 0.312 N |
| 力最大误差 | 2.940 N | 2.672 N |
| TCP平均误差 | 0.191 mm | 0.193 mm |
| TCP最大误差 | 1.081 mm | 1.107 mm |
| 姿态最大误差 | 0.359° | 0.361° |
| 最大电机力矩 | 39.867 Nm | 39.868 Nm |
| 约束处理 | 求解后裁剪 | HQP内部硬约束，无事后裁剪 |

完整自动生成报告：[106_tsid_vs_97d_comparison.md](experiments/ur5_surface_scanning/106_tsid_vs_97d_comparison.md)。

### 107 TSID/HQP 鲁棒性

TSID/HQP在与97E相同的7个完整路径场景中通过6/7。工件X/Z偏移、1°倾斜和低摩擦全部通过；高摩擦 `μ=0.60` 因换行瞬态力误差超阈值而失败，但仍无QP失败、碰撞和电机边界违规。

[查看97D与TSID/HQP逐场景对比报告](experiments/ur5_surface_scanning/results/tsid_robustness/107_tsid_vs_97d_robustness.md)。

## 97E 基准控制器鲁棒性验证

97E 沿用早期 75 系列的受控变量方法：控制器和 97B 学习轨迹保持不变，只改变控制器未知的工件或接触环境。

| 场景 | 结果 | 力平均/最大误差 | TCP最大误差 | 法向修正范围 |
|---|---|---:|---:|---:|
| 基准 | PASS | 0.322 / 2.940 N | 1.081 mm | -0.001～+0.767 mm |
| 工件 Z +1 mm | PASS | 0.315 / 2.656 N | 1.085 mm | -1.003～-0.119 mm |
| 工件 Z -1 mm | PASS | 0.315 / 2.890 N | 1.108 mm | +0.997～+1.656 mm |
| 工件 X +1 mm | PASS | 0.321 / 2.893 N | 1.120 mm | -0.158～+0.996 mm |
| 工件绕 Y +1° | PASS | 0.314 / 2.973 N | 1.073 mm | -1.419～+2.131 mm |
| 低摩擦 μ=0.15 | PASS | 0.322 / 2.940 N | 1.081 mm | -0.001～+0.767 mm |
| 高摩擦 μ=0.60 | FAIL | 0.527 / 7.076 N | 1.476 mm | -0.001～+0.805 mm |

高摩擦场景没有发生碰撞或力矩饱和，但换行瞬态超过当前力误差阈值。这是已记录的控制边界，不会被结果汇总隐藏。

完整结果位于：

```text
experiments/ur5_surface_scanning/results/robustness/
├── 97E_robustness_summary.csv
└── 97E_robustness_summary.npz
```

## 快速运行

```bash
cd ~/ur5_ft300_model_based_control/experiments/ur5_surface_scanning
conda activate ur5-control

# 学习曲面全路径可达性
env -u PYTHONPATH python 97C_check_learned_surface_reachability.py

# 5 N完整扫描
env -u PYTHONPATH python 97D_ur5_learned_surface_5n_scan.py

# 七个完整鲁棒性场景
env -u PYTHONPATH python 97E_ur5_learned_surface_robustness.py

# 从97D结果重新生成README媒体
MUJOCO_GL=egl env -u PYTHONPATH python 98_generate_ur5_scan_demo.py

# TSID模型一致性和MuJoCo姿态闭环
env -u PYTHONPATH python 100_validate_ur5_tsid_model.py
env -u PYTHONPATH python 101_ur5_tsid_mujoco_posture.py

# TSID笛卡尔多任务与HQP硬约束
env -u PYTHONPATH python 102_ur5_tsid_mujoco_se3_hierarchy.py
env -u PYTHONPATH python 103_ur5_tsid_hqp_safety_bounds.py

# TSID/HQP + FT300 5 N接触状态机
env -u PYTHONPATH python 104_ur5_tsid_ft300_5n_contact.py

# TSID/HQP学习曲面完整扫描和基准对比
env -u PYTHONPATH python 105_ur5_tsid_learned_surface_5n_scan.py
env -u PYTHONPATH python 106_compare_97d_vs_tsid_hqp.py

# TSID/HQP七场景鲁棒性验证
env -u PYTHONPATH python 107_ur5_tsid_learned_surface_robustness.py
```

关闭 MuJoCo Viewer 即可退出交互式扫描程序。97E 使用无界面仿真，并自动在完成后退出。

## 74/75 与 97A–97E 的关系

| UR3 早期流程 | UR5 当前流程 |
|---|---|
| 74A 低力扫描学习 | 97A 三维光栅低力探索 |
| 74B 学习路径 5 N 回放 | 97D 学习曲面 5 N 完整扫描 |
| 75 工件/摩擦/工具/噪声鲁棒性 | 97E 工件位姿与摩擦鲁棒性 |
| 约 30.7 mm 路径 | 约 1538.2 mm、961 位姿 |
| 平面内法向估计 | 完整三维曲面重建与 6D 姿态 |

旧实验保留在 `experiments/archive/ur3_ft300_learning/`，用于复现控制器的演进过程。

## 基础示例与 TSID

`examples/` 包含 SO(3)/SE(3)、正运动学、雅可比、wrench 映射、RNEA/CRBA/ABA 和计算力矩控制示例。

UR3 归档目录还保留：

- 关节 PD 与重力补偿
- 笛卡尔阻抗/导纳
- FT300 去零、重力补偿和滤波
- 曲面混合位置/力控制
- TSID/HQP 姿态、SE(3)、约束和接触力实验

## 环境

当前完整 UR5 流程已在以下环境验证：

- Ubuntu 22.04
- Python 3.10
- MuJoCo 3.13.0
- Pinocchio 4.0.0
- TSID 1.10.0
- NumPy

可使用仓库中的 `environment.yml` 创建环境；如果终端已加载 ROS 2 的 Python 路径，运行独立 MuJoCo/Pinocchio 程序时使用 `env -u PYTHONPATH`。

## 注意事项

- 本仓库用于仿真与算法验证。
- 仿真增益不能直接用于真实 UR5。
- 上真实机械臂前必须重新验证关节力矩、速度、碰撞、急停、FT300 量程、控制周期和接触稳定性。
