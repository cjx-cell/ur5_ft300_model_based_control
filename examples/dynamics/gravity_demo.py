import numpy as np
import pinocchio as pin


# ============================================================
# 1. 加载机器人模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

# model：
# 保存机器人固定的结构和动力学参数，例如：
#
# - joint 类型
# - link 质量
# - 惯量矩阵
# - 质心位置
# - joint/link 之间的坐标变换
#
# 后面计算动力学时，这些参数非常重要。
model = pin.buildModelFromUrdf(urdf_path)

# 创建算法计算所需的数据缓存。
data = model.createData()


# ============================================================
# 2. 打印模型中的重力
# ============================================================

# Pinocchio 的 model 中保存了重力加速度。
#
# 对正常机器人模型通常应该类似：
#
# [0, 0, -9.81]
#
# 意思是：
# 世界坐标系 Z 正方向向上，
# 重力沿 Z 负方向。
print("Gravity in model:")
print(model.gravity.linear)


# ============================================================
# 3. 定义机器人当前姿态
# ============================================================

# 获取合法 neutral configuration。
q0 = pin.neutral(model)

# 当前实际希望的六个关节角。
theta = np.deg2rad([
    30.0,      # shoulder_pan
    -45.0,     # shoulder_lift
    60.0,      # elbow
    0.0,       # wrist_1
    20.0,      # wrist_2
    90.0       # wrist_3
])

# 因为：
#
# model.nq = 7
# model.nv = 6
#
# 使用 integrate 把实际六自由度的角度增量
# 转成合法 Pinocchio configuration。
q = pin.integrate(
    model,
    q0,
    theta
)


# ============================================================
# 4. 计算广义重力力矩 g(q)
# ============================================================

# computeGeneralizedGravity：
#
# 输入：
#
# model -> 机器人动力学模型
# data  -> 计算缓存
# q     -> 当前机器人姿态
#
# 输出：
#
# g(q) ∈ R^nv
#
# 你的 UR3：
#
# nv = 6
#
# 所以 gravity_torque 是一个 6 维向量。
gravity_torque = pin.computeGeneralizedGravity(
    model,
    data,
    q
)


# ============================================================
# 5. 打印结果
# ============================================================

np.set_printoptions(
    precision=5,
    suppress=True
)

print("\nJoint angles [deg]:")
print(np.rad2deg(theta))

print("\nGeneralized gravity torque g(q):")
print(gravity_torque)

print("\nEach joint:")

for i, tau_i in enumerate(gravity_torque):
    print(
        f"joint {i + 1}: "
        f"{tau_i: .5f} N*m"
    )

# ============================================================
# 6. 比较不同姿态下的重力力矩
# ============================================================

print("\n\n========== Gravity at Different Poses ==========")


# ------------------------------------------------------------
# Pose A：所有关节 0°
# ------------------------------------------------------------

theta_A = np.deg2rad([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])

q_A = pin.integrate(
    model,
    q0,
    theta_A
)

g_A = pin.computeGeneralizedGravity(
    model,
    data,
    q_A
).copy()


# ------------------------------------------------------------
# Pose B：改变肩关节和肘关节
# ------------------------------------------------------------

theta_B = np.deg2rad([
    0.0,
    -90.0,
    90.0,
    0.0,
    0.0,
    0.0
])

q_B = pin.integrate(
    model,
    q0,
    theta_B
)

g_B = pin.computeGeneralizedGravity(
    model,
    data,
    q_B
).copy()


print("\nPose A:")
print(np.rad2deg(theta_A))

print("Gravity torque:")
print(g_A)


print("\nPose B:")
print(np.rad2deg(theta_B))

print("Gravity torque:")
print(g_B)

# ============================================================
# 7. 用 RNEA 验证 gravity torque
# ============================================================

# 机器人关节速度。
#
# 现在设为全 0：
#
# dq = 0
v = np.zeros(model.nv)


# 机器人关节加速度。
#
# 同样设为 0：
#
# ddq = 0
a = np.zeros(model.nv)


# RNEA：
#
# 输入：
#
# q -> position
# v -> velocity
# a -> acceleration
#
# 输出：
#
# tau -> inverse dynamics torque
tau_rnea = pin.rnea(
    model,
    data,
    q,
    v,
    a
).copy()


print("\n========== RNEA Check ==========")

print("\ng(q):")
print(gravity_torque)

print("\nRNEA(q, 0, 0):")
print(tau_rnea)

print("\nDifference:")
print(tau_rnea - gravity_torque)

print(
    "\nError norm:",
    np.linalg.norm(
        tau_rnea - gravity_torque
    )
)