import numpy as np


def Rx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])


def Rz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])


theta_x = np.deg2rad(90)
theta_z = np.deg2rad(90)

R_xz = Rz(theta_z) @ Rx(theta_x)
R_zx = Rx(theta_x) @ Rz(theta_z)

print("First X, then Z:")
print(R_xz)

print("\nFirst Z, then X:")
print(R_zx)

print("\nAre they equal?")
print(np.allclose(R_xz, R_zx))