import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin



# ============================================================
# 1. Load MuJoCo
# ============================================================


xml_path = Path(__file__).parent / "ur3_converted.xml"


mj_model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)


mj_data = mujoco.MjData(
    mj_model
)



# ============================================================
# 2. Load Pinocchio
# ============================================================


urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"


pin_model = pin.buildModelFromUrdf(
    urdf_path
)


pin_data = pin_model.createData()



print("===================")

print(
    "MuJoCo:",
    mj_model.nq,
    mj_model.nv
)

print(
    "Pinocchio:",
    pin_model.nq,
    pin_model.nv
)

print("===================")



# ============================================================
# 3. MuJoCo q -> Pinocchio q
# ============================================================


def mujoco_to_pinocchio(q_mj):


    q_pin = pin.neutral(
        pin_model
    )


    # 前五轴

    q_pin[:5] = q_mj[:5]


    # wrist3 continuous

    theta6=q_mj[5]


    q_pin[5]=np.cos(theta6)

    q_pin[6]=np.sin(theta6)


    return q_pin



# ============================================================
# 4. Initial pose
# ============================================================


mj_data.qpos[:] = np.deg2rad([

    0,
    -30,
    30,
    0,
    0,
    0

])


mj_data.qvel[:] = 0


mujoco.mj_forward(
    mj_model,
    mj_data
)



# ============================================================
# 5. Target
# ============================================================


q_des=np.deg2rad([

    30,
    -60,
    90,
    -30,
    45,
    0

])



# ============================================================
# 6. Gain
# ============================================================


Kp=np.array([

    80,
    80,
    60,
    40,
    30,
    0

])


Kd=np.array([

    15,
    15,
    12,
    8,
    6,
    0

])



# ============================================================
# 7. Controller
# ============================================================


def controller():



    q=mj_data.qpos.copy()

    dq=mj_data.qvel.copy()



    # -------------------------------
    # error
    # -------------------------------


    e=np.zeros(6)


    e[:5]=np.arctan2(

        np.sin(q_des[:5]-q[:5]),

        np.cos(q_des[:5]-q[:5])

    )



    # 速度误差

    edot=-dq



    # -------------------------------
    # PD -> desired acceleration
    # -------------------------------


    a_cmd=(

        Kp*e

        +

        Kd*edot

    )



    # -------------------------------
    # Pinocchio
    # -------------------------------


    q_pin=mujoco_to_pinocchio(q)



    # velocity

    dq_pin=dq



    # inertia matrix

    M=pin.crba(

        pin_model,

        pin_data,

        q_pin

    )


    M=(M+M.T)/2



    # nonlinear term

    h=pin.nonLinearEffects(

        pin_model,

        pin_data,

        q_pin,

        dq_pin

    )



    # -------------------------------
    # computed torque
    # -------------------------------


    tau=M@a_cmd+h



    # wrist3 off

    tau[5]=0



    # limit

    tau=np.clip(

        tau,

        [-50,-50,-30,-12,-12,0],

        [50,50,30,12,12,0]

    )



    mj_data.ctrl[:]=tau



    return tau,M,h



# ============================================================
# 8. Simulation
# ============================================================


step=0


with mujoco.viewer.launch_passive(

    mj_model,

    mj_data

) as viewer:


    while viewer.is_running():


        tau,M,h=controller()



        mujoco.mj_step(

            mj_model,

            mj_data

        )


        viewer.sync()



        if step%1000==0:


            print("\n=====STATE=====")


            for i in range(6):

                print(

                    i,

                    mj_model.joint(i).name,

                    "q=",

                    mj_data.qpos[i],

                    "dq=",

                    mj_data.qvel[i]

                )


            print("tau=")

            print(tau)



        step+=1


        time.sleep(
            mj_model.opt.timestep
        )