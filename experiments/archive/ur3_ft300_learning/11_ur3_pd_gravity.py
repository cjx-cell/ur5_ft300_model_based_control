import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin



# ============================================================
# 1. Load MuJoCo model
# ============================================================


xml_path = Path(__file__).parent / "ur3_converted.xml"


mj_model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)


mj_data = mujoco.MjData(
    mj_model
)


print("==============================")
print("MuJoCo")
print("nq =", mj_model.nq)
print("nv =", mj_model.nv)



# ============================================================
# 2. Load Pinocchio model
# ============================================================


urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"


pin_model = pin.buildModelFromUrdf(
    urdf_path
)


pin_data = pin_model.createData()


print("Pinocchio")
print("nq =", pin_model.nq)
print("nv =", pin_model.nv)

print("==============================")



# ============================================================
# 3. MuJoCo q -> Pinocchio q
# ============================================================


def mujoco_to_pinocchio(q_mj):


    q_pin = pin.neutral(
        pin_model
    )


    # 前五个普通旋转关节

    q_pin[0] = q_mj[0]
    q_pin[1] = q_mj[1]
    q_pin[2] = q_mj[2]
    q_pin[3] = q_mj[3]
    q_pin[4] = q_mj[4]


    # wrist3 continuous joint
    #
    # Pinocchio:
    #
    # q[5]=cos(theta)
    # q[6]=sin(theta)

    theta6 = q_mj[5]


    q_pin[5] = np.cos(theta6)

    q_pin[6] = np.sin(theta6)


    return q_pin



# ============================================================
# 4. Initial state
# ============================================================


mj_data.qpos[:] = np.deg2rad([

    0,
    -30,
    30,
    0,
    0,
    0

])


# 初始速度清零

mj_data.qvel[:] = 0


mujoco.mj_forward(
    mj_model,
    mj_data
)



# ============================================================
# 5. Target joint position
# ============================================================


q_des = np.deg2rad([

    30,
    -60,
    90,
    -30,
    45,
    0

])



# ============================================================
# 6. Gains
# ============================================================


Kp = np.array([

    40,
    40,
    30,
    20,
    15,
    0

])


Kd = np.array([

    8,
    8,
    6,
    5,
    5,
    0

])



# ============================================================
# 7. Torque limit
# ============================================================


tau_limit = np.array([

    50,
    50,
    30,
    12,
    12,
    0

])



# ============================================================
# 8. Controller
# ============================================================


def controller():



    # 当前状态

    q = mj_data.qpos.copy()

    dq = mj_data.qvel.copy()



    # ========================================================
    # Joint error
    # ========================================================


    error = np.zeros(6)


    # 只控制前5轴

    error[:5] = np.arctan2(

        np.sin(q_des[:5]-q[:5]),

        np.cos(q_des[:5]-q[:5])

    )


    # 第六轴不控制


    error[5]=0



    # ========================================================
    # PD
    # ========================================================


    tau_pd = (

        Kp * error

        -

        Kd * dq

    )



    # ========================================================
    # Gravity compensation
    # ========================================================


    q_pin = mujoco_to_pinocchio(q)


    gravity = pin.computeGeneralizedGravity(

        pin_model,

        pin_data,

        q_pin

    )



    # ========================================================
    # PD + Gravity
    # ========================================================


    tau = tau_pd + gravity



    # ========================================================
    # 强制关闭 wrist3
    #
    # 目的：
    # 验证前5轴控制
    #
    # ========================================================

    tau[5]=0



    tau = np.clip(

        tau,

        -tau_limit,

        tau_limit

    )



    mj_data.ctrl[:] = tau



    return gravity



# ============================================================
# 9. Simulation
# ============================================================


step = 0


with mujoco.viewer.launch_passive(

    mj_model,

    mj_data

) as viewer:



    while viewer.is_running():


        gravity = controller()



        mujoco.mj_step(

            mj_model,

            mj_data

        )


        viewer.sync()



        if step % 1000 == 0:


            print("\n========== STATE ==========")


            for i in range(6):


                print(

                    i,

                    mj_model.joint(i).name,

                    "q=",

                    mj_data.qpos[i],

                    "dq=",

                    mj_data.qvel[i]

                )


            print("\ngravity=")

            print(gravity)



        step += 1


        time.sleep(
            mj_model.opt.timestep
        )