import pinocchio as pin


urdf_path = "//home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"


# 读取 URDF
model = pin.buildModelFromUrdf(urdf_path)

# 为这个 model 创建计算缓存
data = model.createData()

for jid in range(1, model.njoints):
    joint = model.joints[jid]

    print(
        f"id={jid:2d}  "
        f"name={model.names[jid]:25s}  "
        f"type={joint.shortname():30s}  "
        f"nq={joint.nq}  "
        f"nv={joint.nv}  "
        f"idx_q={joint.idx_q}  "
        f"idx_v={joint.idx_v}"
    )

q = pin.neutral(model)

print("Robot name:", model.name)

print("nq =", model.nq)
print("nv =", model.nv)

print("Number of joints =", model.njoints)
print("Number of frames =", len(model.frames))
print("\n===== JOINT DETAILS =====")

print(q)
print("\n===== JOINTS =====")

for i in range(model.njoints):
    print(
        i,
        model.names[i]
    )
print("\n===== FRAMES =====")

for i, frame in enumerate(model.frames):
    print(
        i,
        frame.name,
        frame.type
    )