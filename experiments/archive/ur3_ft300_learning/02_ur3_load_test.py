import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# 1. UR3 URDF 路径
# ============================================================

xml_path = Path(__file__).parent / "ur3_converted.xml"

print("xml path:", xml_path)
print("XML exists:", xml_path.exists())


# ============================================================
# 2. 让 MuJoCo 从 URDF 创建机器人模型
# ============================================================

model = mujoco.MjModel.from_xml_path(str(xml_path))
print("Before actuator:")
print("nu =", model.nu)

# ============================================================
# 3. 创建当前仿真状态 data
# ============================================================

data = mujoco.MjData(model)


# ============================================================
# 4. 打印模型基本信息
# ============================================================

print("========== MuJoCo UR3 ==========")

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

print("\nqpos:")
print(data.qpos)

print("\nqvel:")
print(data.qvel)
print("\n========== Joint Information ==========")

for i in range(model.njnt):

    print(
        i,
        model.joint(i).name
    )


print("\n========== qpos address ==========")


for i in range(model.njnt):

    adr = model.jnt_qposadr[i]

    print(
        model.joint(i).name,
        "qpos index:",
        adr
    )
 