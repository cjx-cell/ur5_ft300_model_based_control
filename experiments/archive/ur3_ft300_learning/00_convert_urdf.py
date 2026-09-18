import mujoco


urdf = "ur3_mujoco.urdf"


model = mujoco.MjModel.from_xml_path(
    urdf
)


xml = mujoco.mj_saveLastXML(
    "ur3_converted.xml",
    model
)

print("saved")