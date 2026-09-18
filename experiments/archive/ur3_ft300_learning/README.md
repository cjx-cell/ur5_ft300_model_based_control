# UR3 早期学习与控制实验归档

本目录保留项目从基础模型控制到未知曲面学习的完整演进过程，文件没有被删除。

## 内容

```text
00–12  模型转换、关节PD、重力补偿、计算力矩
20–32  笛卡尔位置/位姿、阻抗控制
40–54  接触测试、FT300读取、去零、重力补偿、滤波
60–71  导纳与位置/力混合控制
72–73  曲面TCP与未知曲面跟踪实验
74     低力学习 + 学习路径恒力回放
75     74流程对应的鲁棒性实验
80–83  TSID/HQP任务与接触力优化
```

## 74 → 75 基线

74 系列建立了早期的未知曲面工作流：

```text
74A 低力扫描并记录实际TCP
  ↓
拟合学习路径与法向
  ↓
74B 沿学习路径进行5 N回放
```

75 系列保持 74 的学习轨迹不变，只改变隐藏环境或测量条件：

| 文件 | 扰动 |
|---|---|
| `75_ur3_workpiece_offset_robustness.py` | 工件法向安装偏移 |
| `75_ur3_friction_robustness.py` | 接触摩擦变化 |
| `75_ur3_tool_radius_robustness.py` | 探针半径偏差 |
| `75_ur3_scan_noise_robustness_v2.py` | 扫描位置噪声与法向重建 |

当前 UR5 主线中的 `97E_ur5_learned_surface_robustness.py` 沿用了同样的受控变量方法，但验证的是完整 1.54 m 三维扫描路径。

## 运行示例

```bash
cd ~/ur3_ft300_model_based_control/experiments/archive/ur3_ft300_learning
conda activate robot310

env -u PYTHONPATH python 74_ur3_contact_path_learning.py
env -u PYTHONPATH python 74_ur3_force_controlled_replay.py
env -u PYTHONPATH python 75_ur3_workpiece_offset_robustness.py
```

这些脚本主要用于复现实验演进与对照，不再是当前推荐入口。
