import pinocchio as pin


urdf_path="/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"


model = pin.buildModelFromUrdf(
    urdf_path
)


print("nq =",model.nq)
print("nv =",model.nv)



for i in range(model.njoints):

    print(
        i,
        model.names[i],
        model.joints[i].shortname()
    )