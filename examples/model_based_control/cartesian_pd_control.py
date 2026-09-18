import numpy as np
import pinocchio as pin


urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"


model = pin.buildModelFromUrdf(
    urdf_path
)

data = model.createData()


q0 = pin.neutral(model)


theta = np.deg2rad([
    30,
    -45,
    60,
    0,
    20,
    90
])


q = pin.integrate(
    model,
    q0,
    theta
)


# ------------------------------------------------
# Forward Kinematics
# ------------------------------------------------

pin.forwardKinematics(
    model,
    data,
    q
)


pin.updateFramePlacements(
    model,
    data
)



# tool0

ee_id = model.getFrameId(
    "tool0"
)



T = data.oMf[ee_id]



# position

x = T.translation



# rotation

R = T.rotation



print("position:")
print(x)


print("\nrotation:")
print(R)