import numpy as np
import pinocchio as pin

theta = np.deg2rad(90.0)

R = pin.AngleAxis(theta, np.array([0.0, 0.0, 1.0])).matrix()

print(R)