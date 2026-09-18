import numpy as np
import pinocchio as pin


# ============================================================
# 1. 加载机器人
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()


# ============================================================
# 2. 当前姿态 q
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
# 3. 当前速度 v
# ============================================================

v = np.deg2rad([
    5.0,
    -3.0,
    2.0,
    1.0,
    -2.0,
    4.0
])


# ============================================================
# 4. 指定关节力矩 tau
# ============================================================

# 假设六个电机当前输出下面这些力矩。
#
# 单位：
# N*m
tau = np.array([
    1.0,
    -4.0,
    -2.0,
    0.5,
    -0.2,
    0.1
])


# ============================================================
# 5. 使用 ABA 计算正动力学
# ============================================================

# ABA 输入：
#
# q   -> 当前位置
# v   -> 当前速度
# tau -> 当前关节驱动力矩
#
# 输出：
#
# ddq -> 当前关节加速度
#
# 数学对应：
#
# ddq =
#
# M(q)^(-1)
# [
#     tau - h(q,v)
# ]
ddq_aba = pin.aba(
    model,
    data,
    q,
    v,
    tau
).copy()


# ============================================================
# 6. 打印 ABA 结果
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True
)

print("========== ABA Forward Dynamics ==========")

print("\nJoint torque tau:")
print(tau)

print("\nJoint acceleration ddq:")
print(ddq_aba)

# ============================================================
# 7. CRBA 计算质量矩阵 M
# ============================================================

M_upper = pin.crba(
    model,
    data,
    q
).copy()

# 显式补成对称矩阵。
M = (
    np.triu(M_upper)
    +
    np.triu(M_upper, 1).T
)


# ============================================================
# 8. 计算 nonlinear effects h
# ============================================================

h = pin.nonLinearEffects(
    model,
    data,
    q,
    v
).copy()


# ============================================================
# 9. 根据动力学方程手算 ddq
# ============================================================

# 方程：
#
# M ddq + h = tau
#
# 所以：
#
# M ddq = tau - h
#
# 使用：
#
# np.linalg.solve(M, b)
#
# 解线性方程：
#
# M x = b
#
# 不建议写：
#
# inv(M) @ b
ddq_manual = np.linalg.solve(
    M,
    tau - h
)


# ============================================================
# 10. 对比
# ============================================================

print("\n========== Manual Forward Dynamics ==========")

print("\nABA ddq:")
print(ddq_aba)

print("\nSolve(M, tau-h):")
print(ddq_manual)

print("\nDifference:")
print(
    ddq_aba
    -
    ddq_manual
)

print(
    "\nError norm:",
    np.linalg.norm(
        ddq_aba
        -
        ddq_manual
    )
)


# ============================================================
# 11. RNEA -> ABA 闭环验证
# ============================================================

ddq_target = np.array([
    1.0,
    -0.5,
    0.8,
    0.2,
    -0.3,
    0.5
])


# 先通过 inverse dynamics：
#
# q, v, desired ddq
#
# 得到所需 tau。
tau_required = pin.rnea(
    model,
    data,
    q,
    v,
    ddq_target
).copy()


# 再通过 forward dynamics：
#
# q, v, tau
#
# 看机器人产生多少 ddq。
ddq_result = pin.aba(
    model,
    data,
    q,
    v,
    tau_required
).copy()


print("\n========== RNEA -> ABA Check ==========")

print("\nTarget ddq:")
print(ddq_target)

print("\nTorque from RNEA:")
print(tau_required)

print("\nddq from ABA:")
print(ddq_result)

print("\nError:")
print(
    ddq_result - ddq_target
)