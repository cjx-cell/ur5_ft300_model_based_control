import numpy as np
import pinocchio as pin


# ==========================
# 1. 定义旋转矩阵
# ==========================

# 暂时不旋转
R = np.eye(3)

# ==========================
# 2. 定义平移向量
# ==========================

p = np.array([
    0.5,
    0.2,
    0.3
])

# ==========================
# 3. 构造 SE(3)
# ==========================

T = pin.SE3(R, p)

print("SE3:")
print(T)

print("\nRotation:")
print(T.rotation)

print("\nTranslation:")
print(T.translation)

print("\nHomogeneous matrix:")
print(T.homogeneous)