import numpy as np
import pinocchio as pin

theta = np.deg2rad(90.0)

R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])  

p = np.array([1, 0, 0])

T_AB = pin.SE3(R, p)

print(T_AB.homogeneous)