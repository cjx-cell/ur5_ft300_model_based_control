import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer



# ============================================================
# 1. Load MuJoCo model
# ============================================================

# 加载已经添加 actuator 的 MJCF 文件
#
# 这个文件里面包含：
#
# robot body
# joint
# inertia
# actuator
#
# MuJoCo 根据它建立动力学模型。

xml_path = Path(__file__).parent / "ur3_converted.xml"


model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)


# ============================================================
# 2. 创建仿真数据
# ============================================================

# MjData 保存当前时刻状态：
#
# qpos:
#       joint position
#
# qvel:
#       joint velocity
#
# ctrl:
#       actuator input torque
#
data = mujoco.MjData(model)



print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)



# ============================================================
# 3. 查看 joint 映射
# ============================================================

print("\n========== MuJoCo Joint Map ==========")


for i in range(model.njnt):


    print(
        "joint id:",
        i
    )

    print(
        "name:",
        model.joint(i).name
    )


    print(
        "qpos index:",
        model.jnt_qposadr[i]
    )


    print(
        "qvel index:",
        model.jnt_dofadr[i]
    )

    print("-----------------------------")



# ============================================================
# 4. 设置初始姿态
# ============================================================

# MuJoCo 中：
#
# qpos:
# [q1,q2,q3,q4,q5,q6]
#
# 对应：
#
# 0 shoulder_pan
# 1 shoulder_lift
# 2 elbow
# 3 wrist1
# 4 wrist2
# 5 wrist3


data.qpos[:] = np.deg2rad([
    0,
    -30,
    30,
    0,
    0,
    0
])


# 修改 qpos 后必须调用 mj_forward
#
# 作用：
#
# 根据新的关节角
# 更新：
#
# body pose
# geom pose
# sensor
#

mujoco.mj_forward(
    model,
    data
)



# ============================================================
# 5. 设置目标关节角
# ============================================================

# 单位：
# rad


q_des = np.deg2rad([

    30,
    -60,
    90,
    -30,
    45,
    20

])



# ============================================================
# 6. PD 参数
# ============================================================


# 比例增益
#
# 越大：
# 反应越快
# 但是容易振荡


Kp = np.array([

    40,
    40,
    30,
    20,
    15,
    10

])



# 微分增益
#
# 提供阻尼
#
# 抑制振荡


Kd = np.array([

    8,
    8,
    6,
    5,
    5,
    5

])



# ============================================================
# 7. Torque limit
# ============================================================

# 模拟真实电机最大输出力矩
#
# 防止：
#
# tau过大
# 导致机器人飞出去


tau_limit = np.array([

    50,
    50,
    50,
    20,
    20,
    20

])



# ============================================================
# 8. Controller
# ============================================================


def controller(model, data):


    # --------------------------------------------------------
    # 当前关节位置
    # --------------------------------------------------------

    q = data.qpos.copy()



    # 当前关节速度

    dq = data.qvel.copy()



    # --------------------------------------------------------
    # 计算角度误差
    # --------------------------------------------------------

    # 普通：
    #
    # error=q_des-q
    #
    # 对连续旋转关节可能出现：
    #
    # 370deg - 10deg
    #
    # 这种问题
    #
    # atan2(sin,cos)
    # 可以把角度限制到 [-pi,pi]


    error = np.arctan2(

        np.sin(q_des-q),

        np.cos(q_des-q)

    )



    # --------------------------------------------------------
    # PD Controller
    # --------------------------------------------------------

    # tau =
    #
    # Kp*position_error
    #
    # -
    #
    # Kd*velocity
    #
    #
    # 因为目标速度为0


    tau = (

        Kp * error

        -

        Kd * dq

    )



    # --------------------------------------------------------
    # Torque saturation
    # --------------------------------------------------------

    tau = np.clip(

        tau,

        -tau_limit,

        tau_limit

    )



    # --------------------------------------------------------
    # DEBUG:
    #
    # 暂时关闭 wrist3
    #
    # 测试是不是 continuous joint 导致
    #
    # wrist3 index = 5
    # --------------------------------------------------------

    tau[5] = 0



    # 输出给 MuJoCo actuator

    data.ctrl[:] = tau





# ============================================================
# 9. Simulation Loop
# ============================================================


step = 0


with mujoco.viewer.launch_passive(

    model,

    data

) as viewer:



    while viewer.is_running():



        # ============================
        # controller
        # ============================

        controller(

            model,

            data

        )



        # ============================
        # physics update
        # ============================

        mujoco.mj_step(

            model,

            data

        )



        # ============================
        # viewer update
        # ============================

        viewer.sync()



        # ============================
        # 每1000步打印一次状态
        # ============================

        if step % 1000 == 0:


            print("\n========== State ==========")


            for i in range(6):


                print(

                    i,

                    model.joint(i).name,

                    "q=",

                    data.qpos[i],

                    "dq=",

                    data.qvel[i]

                )



        step += 1



        time.sleep(

            model.opt.timestep

        )
for i in range(model.nu):

    print(
        i,
        model.actuator(i).name,
        model.actuator_trnid[i]
    )


print("\n========== Actuator Map ==========")


for i in range(model.nu):

    print(
        "actuator",
        i,
        "name:",
        model.actuator(i).name,
        "joint id:",
        model.actuator_trnid[i][0]
    )