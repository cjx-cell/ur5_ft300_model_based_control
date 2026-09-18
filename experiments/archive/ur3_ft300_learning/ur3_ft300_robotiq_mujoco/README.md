# UR3 + FT300 + Robotiq 2F-85 MuJoCo 模型

本目录包含从 ROS/Gazebo Xacro 独立转换得到的 MuJoCo 模型。转换过程只读取原始 ROS 文件，不会修改原始 Xacro、URDF 或软件包。

## 文件说明

- `ur3_ft300_robotiq.xml`：最终 MuJoCo 模型，可以直接加载使用。
- `ur3_ft300_robotiq_resolved.urdf`：已经展开 Xacro，并把 ROS 网格路径转换为绝对路径的 URDF。
- `ur3_ft300_robotiq_raw.xml`：MuJoCo 直接导入 URDF 后得到的中间 MJCF，尚未加入 actuator、传感器和碰撞修正，不建议直接使用。
- `convert_model.py`：完整且可重复执行的 Xacro 到 MJCF 转换脚本。
- `view_model.py`：加载最终模型，在 Gazebo home 位姿下使用关节 PD 和 MuJoCo 重力补偿保持机械臂。

## 直接查看最终模型

在仓库根目录运行：

```bash
conda activate ur3-control

python experiments/archive/ur3_ft300_learning/ur3_ft300_robotiq_mujoco/view_model.py
```

也可以直接在自己的 Python 程序中加载：

```python
from pathlib import Path
import mujoco

model_path = Path(
    "experiments/archive/ur3_ft300_learning/"
    "ur3_ft300_robotiq_mujoco/"
    "ur3_ft300_robotiq.xml"
).resolve()

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
```

## 加载 home 初始姿态

模型包含名为 `home` 的 keyframe：

```python
home_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)
```

home 姿态中的 UR3 关节角为：

```text
shoulder_pan_joint  = -1.2540 rad
shoulder_lift_joint = -1.5707 rad
elbow_joint         =  1.5707 rad
wrist_1_joint       = -1.5707 rad
wrist_2_joint       = -1.5707 rad
wrist_3_joint       =  0.0000 rad
```

## 控制输入排列

模型共有 7 个 actuator，`data.ctrl` 排列如下：

```text
ctrl[0]  shoulder_pan_joint 力矩，单位 N·m
ctrl[1]  shoulder_lift_joint 力矩，单位 N·m
ctrl[2]  elbow_joint 力矩，单位 N·m
ctrl[3]  wrist_1_joint 力矩，单位 N·m
ctrl[4]  wrist_2_joint 力矩，单位 N·m
ctrl[5]  wrist_3_joint 力矩，单位 N·m
ctrl[6]  Robotiq 主关节目标位置，单位 rad
```

UR3 六轴的控制范围为：

```text
shoulder_pan_joint   ±56 N·m
shoulder_lift_joint  ±56 N·m
elbow_joint          ±28 N·m
wrist_1_joint        ±12 N·m
wrist_2_joint        ±12 N·m
wrist_3_joint        ±12 N·m
```

夹爪只驱动主关节 `robotiq_85_left_knuckle_joint`，其余五个关节通过从 URDF mimic 关系转换得到的 equality 约束联动。目标位置范围为 `-0.01～0.8 rad`。

```python
# 打开夹爪
data.ctrl[6] = 0.0

# 闭合夹爪
data.ctrl[6] = 0.4
```

## FT300 数据

模型包含两个原生 MuJoCo 传感器：

```text
ft300_force   3维力
ft300_torque  3维力矩
```

读取方法：

```python
force = data.sensordata[0:3].copy()
torque = data.sensordata[3:6].copy()
```

数据在 `ft300_site` 坐标系中表达。

## 相机

模型包含两台 MuJoCo 相机：

```text
wrist_camera   腕部相机
global_camera  全局相机
```

可以通过名称取得相机 ID：

```python
wrist_camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera"
)
global_camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "global_camera"
)
```

## 重新生成模型

原始 Xacro 或网格更新后，在仓库根目录运行：

```bash
conda activate ur3-control

python experiments/archive/ur3_ft300_learning/ur3_ft300_robotiq_mujoco/convert_model.py
```

转换脚本会重新生成：

```text
ur3_ft300_robotiq_resolved.urdf
ur3_ft300_robotiq_raw.xml
ur3_ft300_robotiq.xml
```

并自动检查模型维度、actuator、传感器、相机、五个夹爪 mimic 约束和 home 姿态下的意外碰撞。

## 转换模型的主要处理

- 保留全部 50 个 URDF body，避免 `tool0`、FT300 和相机坐标系被合并。
- 将 ROS `package://` 和 `file://` 网格路径转换为绝对路径。
- 删除 MuJoCo 不使用的 Gazebo 和 `ros2_control` 标签。
- 添加六个 UR3 力矩 actuator 和一个夹爪位置 actuator。
- 将五个夹爪 mimic 关系转换为 equality 约束并提高约束精度。
- 添加 MuJoCo 原生 FT300 force/torque sensor。
- 添加腕部相机和全局相机。
- 添加关节 damping、armature、摩擦和接触参数。
- 排除 base/shoulder 以及夹爪内部的错误自碰撞。
- 添加与 Gazebo 配置一致的 `home` keyframe。

## 视觉网格说明

当前安装的 MuJoCo 不支持源模型使用的 Collada `.dae` 视觉网格。因此最终模型使用 STL 和 primitive 碰撞几何进行物理计算和显示，并为两台 RealSense 添加了简化相机外壳。

这一处理不会改变机械臂和夹爪的运动学、惯量、关节约束、FT300 测量或相机坐标系。
