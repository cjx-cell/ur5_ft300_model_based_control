import pinocchio as pin
import numpy as np


urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

# ==============================
# 1. Load model
# ==============================

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

print("Robot:", model.name)
print("nq:", model.nq)
print("nv:", model.nv)


# ==============================
# 2. Set joint angles
# ==============================

q0 = pin.neutral(model)

theta = np.deg2rad([
    30.0,     # shoulder_pan
    -45.0,    # shoulder_lift
    60.0,     # elbow
    0.0,      # wrist_1
    20.0,     # wrist_2
    90.0      # wrist_3
])

q = pin.integrate(model, q0, theta)

print("\nJoint angles [deg]:")
print(np.rad2deg(theta))

print("\nPinocchio q:")
print(q)


# ==============================
# 3. Forward kinematics
# ==============================

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)


# ==============================
# 4. End-effector
# ==============================

ee_name = "tool0"
ee_id = model.getFrameId(ee_name)

T_0E = data.oMf[ee_id]


print("\n===== TOOL0 =====")

print("\nPosition:")
print(T_0E.translation)

print("\nRotation:")
print(T_0E.rotation)

print("\nHomogeneous transform:")
print(T_0E.homogeneous)