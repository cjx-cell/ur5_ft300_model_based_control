import numpy as np
import pinocchio as pin


# World -> Base
R_WB = np.eye(3)
p_WB = np.array([1.0, 0.0, 0.0])

T_WB = pin.SE3(R_WB, p_WB)


# Base -> End Effector
R_BE = np.eye(3)
p_BE = np.array([0.5, 0.0, 0.3])

T_BE = pin.SE3(R_BE, p_BE)


# World -> End Effector
T_WE = T_WB * T_BE


print("T_WB:")
print(T_WB.homogeneous)

print("\nT_BE:")
print(T_BE.homogeneous)

print("\nT_WE:")
print(T_WE.homogeneous)

print("\nEnd-effector position in world:")
print(T_WE.translation)