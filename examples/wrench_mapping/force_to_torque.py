import numpy as np
import pinocchio as pin


# ============================================================
# 1. 读取机器人模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

# model 保存机器人固定结构：
# joint、link、质量、惯量、坐标关系等等。
model = pin.buildModelFromUrdf(urdf_path)

# data 保存算法运行过程中产生的结果和缓存。
data = model.createData()


# ============================================================
# 2. 指定我们研究的末端 Frame
# ============================================================

# 我们希望研究 tool0：
#
# tool0 受到一个笛卡尔空间的力之后，
# 各个关节应该对应多少关节力矩。
ee_name = "tool0"

# frame name -> Pinocchio 内部整数 ID
ee_id = model.getFrameId(ee_name)


# ============================================================
# 3. 设置当前机器人姿态
# ============================================================

# 合法的 neutral configuration
q0 = pin.neutral(model)

# 六个实际关节角，单位先写 degree，
# 再通过 deg2rad 转换为 rad。
theta = np.deg2rad([
    30.0,      # shoulder_pan
    -45.0,     # shoulder_lift
    60.0,      # elbow
    0.0,       # wrist_1
    20.0,      # wrist_2
    90.0       # wrist_3
])

# 你的机器人：
#
# nq = 7
# nv = 6
#
# 所以利用 integrate 将 6 维关节增量
# 转换成合法的 7 维 configuration。
q = pin.integrate(
    model,
    q0,
    theta
)


# ============================================================
# 4. 计算 tool0 Jacobian
# ============================================================

# 我们使用 LOCAL_WORLD_ALIGNED：
#
# - Jacobian 的作用点位于 tool0 原点
# - XYZ 方向和世界坐标系平行
#
# 所以下面定义的力 F 也必须使用相同表达方式。
J = pin.computeFrameJacobian(
    model,
    data,
    q,
    ee_id,
    pin.LOCAL_WORLD_ALIGNED
)


# ============================================================
# 5. 定义 tool0 的目标 Wrench
# ============================================================

# 六维 Wrench：
#
# F =
#
# [ Fx ]
# [ Fy ]
# [ Fz ]
# [ Mx ]
# [ My ]
# [ Mz ]
#
# 前三维：力，单位 N
# 后三维：力矩，单位 N·m
#
# 现在我们只希望 tool0 沿世界 Z 正方向
# 产生 10 N 的力。
wrench = np.array([
    0.0,       # Fx [N]
    0.0,       # Fy [N]
    10.0,      # Fz [N]
    0.0,       # Mx [N*m]
    0.0,       # My [N*m]
    0.0        # Mz [N*m]
])


# ============================================================
# 6. Wrench -> Joint Torque
# ============================================================

# 核心公式：
#
# tau = J^T F
#
# J：
# 6 × 6
#
# J.T：
# 6 × 6
#
# wrench：
# 6 × 1
#
# 最终：
#
# tau：
# 6 × 1
#
# 对应 UR3 六个关节的广义力矩。
tau_task = J.T @ wrench


# ============================================================
# 7. 打印结果
# ============================================================

np.set_printoptions(
    precision=5,
    suppress=True
)

print("========== Jacobian ==========")
print(J)

print("\n========== Desired Wrench ==========")
print(wrench)

print("\n========== Joint Torque ==========")
print(tau_task)

print("\nEach joint torque:")

for i, tau_i in enumerate(tau_task):
    print(
        f"joint {i + 1}: "
        f"{tau_i: .5f} N*m"
    )

# ============================================================
# 8. 验证功率一致性
# ============================================================

# 随便设置一个当前关节速度。
dq = np.deg2rad([
    5.0,
    -3.0,
    2.0,
    1.0,
    -2.0,
    4.0
])

# 根据 Jacobian：
#
# V = J dq
#
# 得到 tool0 空间速度。
V = J @ dq


# ------------------------------------------------------------
# Cartesian side power
# ------------------------------------------------------------

# 末端功率：
#
# P_cartesian = F^T V
#
# 力 N × 线速度 m/s
# +
# 力矩 N*m × 角速度 rad/s
#
# 最终单位都是 W。
P_cartesian = wrench @ V


# ------------------------------------------------------------
# Joint side power
# ------------------------------------------------------------

# 关节功率：
#
# P_joint = tau^T dq
#
# N*m × rad/s
#
# 单位 W。
P_joint = tau_task @ dq


print("\n========== Power Verification ==========")

print("Cartesian power:")
print(P_cartesian, "W")

print("\nJoint power:")
print(P_joint, "W")

print("\nPower error:")
print(P_cartesian - P_joint)