import numpy as np
import pinocchio as pin


# ============================================================
# 1. 加载机器人
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

# 从 URDF 创建动力学模型。
model = pin.buildModelFromUrdf(urdf_path)

# 创建算法计算缓存。
data = model.createData()


# ============================================================
# 2. 设置机器人当前姿态 q
# ============================================================

# 获得合法的 neutral configuration。
q0 = pin.neutral(model)

# 六个实际关节角。
theta = np.deg2rad([
    30.0,
    -45.0,
    60.0,
    0.0,
    20.0,
    90.0
])

# 由于：
#
# nq = 7
# nv = 6
#
# 所以使用 integrate 得到合法 configuration。
q = pin.integrate(
    model,
    q0,
    theta
)


# ============================================================
# 3. 使用 CRBA 计算质量矩阵 M(q)
# ============================================================

# pin.crba：
#
# 输入：
#   model -> 机器人模型
#   data  -> 计算缓存
#   q     -> 当前姿态
#
# 输出：
#   Joint Space Inertia Matrix
#
# 对 UR3：
#
# M ∈ R^(6×6)
#
M_upper = pin.crba(
    model,
    data,
    q
).copy()


# ============================================================
# 4. 把 CRBA 结果显式补成对称矩阵
# ============================================================

# Pinocchio 的 CRBA 算法定义只保证计算质量矩阵上三角部分。
#
# 物理上的质量矩阵满足：
#
#       M = M^T
#
# 所以我们把上三角复制到下三角。
#
# np.triu(M_upper)
#
# 提取：
#
# [x x x]
# [0 x x]
# [0 0 x]
#
upper = np.triu(M_upper)

# np.triu(M_upper, 1)
#
# 参数 1 表示：
# 不包含主对角线，只取严格上三角。
#
# 再 .T 就得到严格下三角。
lower = np.triu(M_upper, 1).T

# 得到完整对称矩阵：
M = upper + lower


# ============================================================
# 5. 打印结果
# ============================================================

np.set_printoptions(
    precision=5,
    suppress=True
)

print("Joint angles [deg]:")
print(np.rad2deg(theta))

print("\n========== Mass Matrix M(q) ==========")
print(M)

print("\nShape:")
print(M.shape)

# ============================================================
# 6. 检查质量矩阵是否对称
# ============================================================

symmetry_error = M - M.T

print("\n========== Symmetry Check ==========")

print("M - M.T:")
print(symmetry_error)

print(
    "\nSymmetry error norm:",
    np.linalg.norm(symmetry_error)
)

# ============================================================
# 7. 检查正定性
# ============================================================

# 对对称矩阵使用 eigvalsh。
#
# eigvalsh 专门用于 Hermitian / symmetric matrix，
# 比普通 eigvals 更合适。
eigenvalues = np.linalg.eigvalsh(M)

print("\n========== Eigenvalues ==========")
print(eigenvalues)

print(
    "\nAll eigenvalues > 0 ?",
    np.all(eigenvalues > 0)
)

# ============================================================
# 8. 计算关节运动对应的动能
# ============================================================

dq = np.deg2rad([
    5.0,
    -3.0,
    2.0,
    1.0,
    -2.0,
    4.0
])

kinetic_energy = 0.5 * dq @ M @ dq

print("\n========== Kinetic Energy ==========")

print("dq:")
print(dq)

print(
    "\nKinetic energy:",
    kinetic_energy,
    "J"
)

# ============================================================
# 9. 定义关节加速度 ddq
# ============================================================

# 我们人为指定机器人当前希望产生的关节加速度。
#
# 单位：
# rad/s^2
ddq = np.array([
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0
])


# ============================================================
# 10. 计算 M(q) * ddq
# ============================================================

# 这一项表示纯惯性产生的关节力矩。
tau_inertia = M @ ddq


# ============================================================
# 11. 计算重力项 g(q)
# ============================================================

gravity = pin.computeGeneralizedGravity(
    model,
    data,
    q
).copy()


# ============================================================
# 12. 用 RNEA 计算完整逆动力学
# ============================================================

# 当前假设关节速度：
#
# dq = 0
#
# 所以科氏力和离心力项消失。
v_zero = np.zeros(model.nv)


# RNEA：
#
# tau =
#
# M(q) ddq
# +
# C(q,dq)dq
# +
# g(q)
#
# 由于 dq=0：
#
# tau =
#
# M(q)ddq + g(q)
tau_rnea = pin.rnea(
    model,
    data,
    q,
    v_zero,
    ddq
).copy()


# ============================================================
# 13. 从 RNEA 中减去 gravity
# ============================================================

tau_rnea_without_gravity = (
    tau_rnea
    -
    gravity
)


# ============================================================
# 14. 对比
# ============================================================

print("\n========== Inertia Term Verification ==========")

print("\nM(q) @ ddq:")
print(tau_inertia)

print("\nRNEA(q, 0, ddq) - g(q):")
print(tau_rnea_without_gravity)

print("\nDifference:")
print(
    tau_inertia
    -
    tau_rnea_without_gravity
)

print(
    "\nError norm:",
    np.linalg.norm(
        tau_inertia
        -
        tau_rnea_without_gravity
    )
)


print("M second column:")
print(M[:, 1])

print("M @ ddq:")
print(M @ ddq)