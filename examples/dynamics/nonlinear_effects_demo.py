import numpy as np
import pinocchio as pin


# ============================================================
# 1. 加载机器人模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()


# ============================================================
# 2. 设置机器人当前姿态 q
# ============================================================

q0 = pin.neutral(model)

theta = np.deg2rad([
    30.0,
    -45.0,
    60.0,
    0.0,
    20.0,
    90.0
])

q = pin.integrate(
    model,
    q0,
    theta
)


# ============================================================
# 3. 设置当前关节速度 v = dq
# ============================================================

# 注意：
#
# 这里不是“角度”，而是“角速度”。
#
# 单位：
# rad/s
#
# nv = 6
#
# 所以这里必须是 6 维。
v = np.zeros(model.nv)


# ============================================================
# 4. 计算重力项 g(q)
# ============================================================

gravity = pin.computeGeneralizedGravity(
    model,
    data,
    q
).copy()


# ============================================================
# 5. 计算 nonlinear effects h(q, v)
# ============================================================

# Pinocchio：
#
# h(q,v)
# =
# C(q,v)v
# +
# g(q)
#
# 它把：
#
# - Coriolis
# - centrifugal
# - gravity
#
# 合成一个广义力向量。
h = pin.nonLinearEffects(
    model,
    data,
    q,
    v
).copy()


# ============================================================
# 6. 从 h 中减去 gravity
# ============================================================

# 因为：
#
# h = Cv + g
#
# 所以：
#
# Cv = h - g
#
# 这里的变量 coriolis_centrifugal
# 实际表示：
#
# C(q,v) @ v
#
# 而不是矩阵 C 本身。
coriolis_centrifugal = (
    h
    -
    gravity
)


# ============================================================
# 7. 打印结果
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True
)

print("========== q ==========")
print(q)

print("\n========== v = dq [rad/s] ==========")
print(v)

print("\n========== g(q) ==========")
print(gravity)

print("\n========== h(q,v) ==========")
print(h)

print("\n========== C(q,v)v ==========")
print(coriolis_centrifugal)

# ============================================================
# 8. 用 RNEA 验证 h(q,v)
# ============================================================

# 关节加速度设为 0。
a_zero = np.zeros(model.nv)


# RNEA：
#
# tau =
#
# M(q)a
# +
# C(q,v)v
# +
# g(q)
#
# 当前 a=0：
#
# tau =
#
# C(q,v)v + g(q)
#
#      =
#
# h(q,v)
tau_rnea = pin.rnea(
    model,
    data,
    q,
    v,
    a_zero
).copy()


print("\n========== RNEA Verification ==========")

print("\nh(q,v):")
print(h)

print("\nRNEA(q,v,0):")
print(tau_rnea)

print("\nDifference:")
print(
    h - tau_rnea
)

print(
    "\nError norm:",
    np.linalg.norm(
        h - tau_rnea
    )
)