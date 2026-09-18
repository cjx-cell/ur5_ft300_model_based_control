import numpy as np
import pinocchio as pin


# ============================================================
# 1. URDF 文件路径
# ============================================================

# URDF 描述了机器人的：
# - link
# - joint
# - 质量
# - 惯量
# - joint axis
# - 坐标关系
#
# Pinocchio 会读取这个文件并建立机器人模型。
urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"


# ============================================================
# 2. 从 URDF 创建 Pinocchio Model
# ============================================================

# model：
# 保存“机器人本身的信息”。
#
# 例如：
# model.nq
# model.nv
# model.names
# model.frames
# model.joints
#
# 注意：
# model 本身一般不保存某一个具体 q 下的计算结果。
model = pin.buildModelFromUrdf(urdf_path)


# ============================================================
# 3. 创建 Data
# ============================================================

# data：
# Pinocchio 运行 FK、Jacobian、Dynamics 等算法时，
# 用来保存计算结果和中间缓存。
#
# 可以理解：
#
# model = 机器人结构
# data  = 这次计算得到的结果
#
data = model.createData()


# ============================================================
# 4. 获取末端 Frame ID
# ============================================================

# 我们之后要计算 tool0 的 Jacobian。
#
# Pinocchio 内部不是每次都通过字符串 "tool0" 查找，
# 而是先把字符串转换成一个整数 ID。
#
# ee_id 可能比如是 23。
#
# 后面直接用 ID 查数据速度更快。
ee_name = "wrist_1_joint"
ee_id = model.getFrameId(ee_name)


# ============================================================
# 5. 获取机器人的 neutral configuration
# ============================================================

# 对普通 revolute joint：
# neutral 就是 theta = 0。
#
# 但你的第六轴是 continuous joint：
#
# q6 = [cos(theta6), sin(theta6)]
#
# 所以它的 neutral 是：
#
# [cos(0), sin(0)] = [1, 0]
#
# 因此不能简单使用：
#
# q0 = np.zeros(model.nq)
#
# 而应该用 Pinocchio 提供的合法 configuration。
q0 = pin.neutral(model)


# ============================================================
# 6. 定义我们想要的 6 个实际关节角
# ============================================================

# np.deg2rad：
# 把角度 degree 转成 radian。
#
# 因为机器人算法和 Pinocchio 都使用 rad。
theta = np.deg2rad([
    30.0,     # joint 1
    -45.0,    # joint 2
    60.0,     # joint 3
    0.0,      # joint 4
    20.0,     # joint 5
    90.0      # joint 6
])


# ============================================================
# 7. 使用 integrate 得到合法的 configuration q
# ============================================================

# 这一步非常重要。
#
# theta 是 6 维：
#
# theta ∈ R^6
#
# 因为机器人实际上只有 6 个自由度。
#
# 但你的 configuration q 是 7 维：
#
# q ∈ R^7
#
# 因为第六轴 continuous joint 用两个数：
#
# [cos(theta6), sin(theta6)]
#
# pin.integrate(model, q0, theta)
#
# 可以理解成：
#
# “从 q0 出发，沿着 theta 这个关节增量移动”
#
# Pinocchio 会自动正确处理：
# - 普通 revolute joint
# - continuous joint
# - quaternion
# - free-flyer
#
# 所以不要手动乱改 q。
q = pin.integrate(model, q0, theta)


# ============================================================
# 8. 计算 tool0 的 Jacobian
# ============================================================

# computeFrameJacobian：
#
# 输入：
# model       -> 机器人模型
# data        -> 计算缓存
# q           -> 当前 configuration
# ee_id       -> 想求哪个 frame 的 Jacobian
# reference   -> Jacobian 用哪个坐标系表达
#
# 输出：
#
# J ∈ R^(6 × nv)
#
# 当前：
#
# nv = 6
#
# 所以：
#
# J.shape = (6,6)
#
J = pin.computeFrameJacobian(
    model,
    data,
    q,
    ee_id,
    pin.LOCAL_WORLD_ALIGNED
)


# ============================================================
# 9. 打印 Jacobian
# ============================================================

# 设置 numpy 输出格式：
#
# precision=5
# 小数点后显示 5 位。
#
# suppress=True
# 很小的数尽量不显示成科学计数法。
np.set_printoptions(
    precision=5,
    suppress=True
)

print("Jacobian:")
print(J)


# ============================================================
# 10. 设置关节速度 dq
# ============================================================

# dq 是关节速度：
#
# dq = [
#   dq1,
#   dq2,
#   ...
#   dq6
# ]
#
# 单位 rad/s。
#
# 注意：
#
# dq 维度跟 nv 一样。
#
# 所以是 6 维，不是 7 维。
dq = np.deg2rad([
    0.0,
    0.0,
    0.0,
    0.0,
    100.0,
    200.0
])


# ============================================================
# 11. Jacobian × 关节速度
# ============================================================

# 数学公式：
#
# V = J(q) dq
#
# J：6×6
#
# dq：6×1
#
# 所以：
#
# V：6×1
#
# V 是 tool0 的空间速度。
V = J @ dq


# ============================================================
# 12. 把空间速度拆成线速度和角速度
# ============================================================

# Pinocchio Jacobian 的排列是：
#
# V =
# [
#   vx
#   vy
#   vz
#   wx
#   wy
#   wz
# ]
#
# Python 切片：
#
# V[:3]
#
# 表示：
#
# V[0], V[1], V[2]
#
v_linear = V[:3]


# V[3:]
#
# 表示从 index=3 一直到最后：
#
# V[3], V[4], V[5]
omega = V[3:]


print("\nLinear velocity:")
print(v_linear)

print("\nAngular velocity:")
print(omega)