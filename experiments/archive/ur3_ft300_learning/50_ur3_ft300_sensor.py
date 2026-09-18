import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 文件路径
# ============================================================
#
# 原始、已经验证好的 MuJoCo 模型保持不动。
#
# 本脚本自动生成：
#
# ur3_ft300_scene.xml
#
# 专门用于：
#
# - 墙体接触
# - FT300 六维力/力矩传感器
# ============================================================

work_dir = Path(__file__).parent

base_xml_path = (
    work_dir
    /
    "ur3_converted.xml"
)

ft300_xml_path = (
    work_dir
    /
    "ur3_ft300_scene.xml"
)


# ============================================================
# 2. 加载 Pinocchio 模型
# ============================================================

urdf_path = (
    "/home/ubuntu/ur3_ft300_ws/"
    "ur3_generated.urdf"
)

pin_model = pin.buildModelFromUrdf(
    urdf_path
)

pin_data = pin_model.createData()


# tool0 frame

ee_id = pin_model.getFrameId(
    "tool0"
)


# ============================================================
# 3. MuJoCo q -> Pinocchio q
# ============================================================

def mujoco_to_pinocchio(q_mj):

    q_pin = pin.neutral(
        pin_model
    )

    # 前5轴直接对应
    q_pin[:5] = q_mj[:5]

    # 第6轴是 JointModelRUBZ
    #
    # configuration：
    #
    # [cos(theta6), sin(theta6)]

    theta6 = q_mj[5]

    q_pin[5] = np.cos(theta6)
    q_pin[6] = np.sin(theta6)

    return q_pin


# ============================================================
# 4. 初始关节姿态
# ============================================================

q_initial_mj = np.deg2rad([
    30.0,
    -60.0,
    90.0,
    -30.0,
    45.0,
    0.0
])


# ============================================================
# 5. 先加载原始 MuJoCo 模型
# ============================================================
#
# 我们需要计算：
#
# tool0 相对于 wrist_3_link 的位姿。
#
# 然后把虚拟 FT300 body 准确安装到 tool0。
# ============================================================

base_model = mujoco.MjModel.from_xml_path(
    str(base_xml_path)
)

base_data = mujoco.MjData(
    base_model
)


base_data.qpos[:] = q_initial_mj
base_data.qvel[:] = 0.0


mujoco.mj_forward(
    base_model,
    base_data
)


# ============================================================
# 6. Pinocchio 获取 tool0 WORLD pose
# ============================================================

q_initial_pin = mujoco_to_pinocchio(
    q_initial_mj
)


pin.forwardKinematics(
    pin_model,
    pin_data,
    q_initial_pin
)

pin.updateFramePlacements(
    pin_model,
    pin_data
)


T_tool_initial = (
    pin_data.oMf[ee_id]
)


p_tool_initial = (
    T_tool_initial
    .translation
    .copy()
)


R_tool_initial = (
    T_tool_initial
    .rotation
    .copy()
)


# ============================================================
# 7. MuJoCo 获取 wrist_3_link WORLD pose
# ============================================================

wrist_body_id = mujoco.mj_name2id(
    base_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "wrist_3_link"
)


if wrist_body_id == -1:

    raise RuntimeError(
        "找不到 wrist_3_link"
    )


p_wrist_world = (
    base_data
    .xpos[wrist_body_id]
    .copy()
)


R_wrist_world = (
    base_data
    .xmat[wrist_body_id]
    .reshape(3, 3)
    .copy()
)


# ============================================================
# 8. tool0 相对于 wrist_3_link 的局部位姿
# ============================================================
#
# WORLD位置：
#
# p_tool =
#
# p_wrist
#
# +
#
# R_wrist * p_local
#
#
# 所以：
#
# p_local =
#
# R_wrist^T
#
# *
#
# (p_tool - p_wrist)
# ============================================================

p_tool_local = (

    R_wrist_world.T

    @

    (
        p_tool_initial
        -
        p_wrist_world
    )
)


# ============================================================
# 9. tool0 相对于 wrist 的旋转
# ============================================================
#
# R_tool_world =
#
# R_wrist_world
#
# *
#
# R_tool_local
#
#
# 所以：
#
# R_tool_local =
#
# R_wrist_world^T
#
# *
#
# R_tool_world
# ============================================================

R_tool_local = (

    R_wrist_world.T

    @

    R_tool_initial

)


# ============================================================
# 10. Rotation Matrix -> MuJoCo Quaternion
# ============================================================
#
# MuJoCo XML body orientation 使用 quaternion。
#
# mju_mat2Quat：
#
# rotation matrix
#
#       ↓
#
# quaternion
#
# MuJoCo quaternion 顺序：
#
# [w, x, y, z]
# ============================================================

tool_local_quat = np.zeros(
    4
)


mujoco.mju_mat2Quat(

    tool_local_quat,

    R_tool_local.reshape(9)

)


# ============================================================
# 11. 修改 XML
# ============================================================

tree = ET.parse(
    base_xml_path
)

root = tree.getroot()


# 找到 wrist_3_link

wrist_body_xml = root.find(
    ".//body[@name='wrist_3_link']"
)


if wrist_body_xml is None:

    raise RuntimeError(
        "XML 中找不到 wrist_3_link"
    )


# ============================================================
# 12. 创建 FT300 dummy body
# ============================================================
#
# 非常重要：
#
# FT300 body 没有 joint。
#
# 所以 MuJoCo 会把它视为：
#
# welded body
#
# 即刚性连接在 wrist_3_link 上。
#
#
# Force/Torque sensor 就测：
#
# FT300 body
#
# 和
#
# wrist_3_link
#
# 之间传递的力。
# ============================================================

ft300_body = ET.SubElement(

    wrist_body_xml,

    "body",

    {
        "name": "ft300_body",

        "pos":
            f"{p_tool_local[0]} "
            f"{p_tool_local[1]} "
            f"{p_tool_local[2]}",

        "quat":
            f"{tool_local_quat[0]} "
            f"{tool_local_quat[1]} "
            f"{tool_local_quat[2]} "
            f"{tool_local_quat[3]}"
    }

)


# ============================================================
# 13. FT300 sensor site
# ============================================================
#
# site 是 MuJoCo 中的“参考坐标系”。
#
# 它：
#
# - 不参与碰撞
# - 不增加质量
# - 可以安装传感器
#
#
# FT300 输出的：
#
# Fx Fy Fz
# Mx My Mz
#
# 都是在这个 site 坐标系表达。
# ============================================================

ET.SubElement(

    ft300_body,

    "site",

    {
        "name": "ft300_site",

        "pos": "0 0 0",

        "size": "0.008",

        "rgba": "0.1 1.0 0.1 1"
    }

)


# ============================================================
# 14. TCP collision probe
# ============================================================
#
# 接触球放在 FT300 body 上。
#
# 这样墙产生的接触力必须经过：
#
# probe
#   ↓
# FT300 body
#   ↓
# wrist_3_link
#
# 因此 FT300 sensor 可以测到它。
# ============================================================

ET.SubElement(

    ft300_body,

    "geom",

    {
        "name": "tcp_probe",

        "type": "sphere",

        "pos": "0 0 0",

        "size": "0.015",

        # 不改变原机器人动力学模型
        "mass": "0",

        # 只和 wall 的 collision bit 发生碰撞
        "contype": "2",

        "conaffinity": "4",

        "condim": "3",

        "friction": "0.8 0.005 0.0001",

        "rgba": "0.9 0.2 0.2 1"
    }

)


# ============================================================
# 15. 添加墙
# ============================================================

worldbody_xml = root.find(
    "worldbody"
)


if worldbody_xml is None:

    raise RuntimeError(
        "XML 中找不到 worldbody"
    )


# 墙中心：
#
# TCP初始位置 +X 60 mm
#
# 墙厚20 mm
#
# probe半径15 mm
#
# 所以大约 +35 mm 时开始接触。

wall_position = (

    p_tool_initial

    +

    np.array([
        0.060,
        0.0,
        0.0
    ])

)


ET.SubElement(

    worldbody_xml,

    "geom",

    {
        "name": "contact_wall",

        "type": "box",

        "pos":
            f"{wall_position[0]} "
            f"{wall_position[1]} "
            f"{wall_position[2]}",

        "size":
            "0.010 0.060 0.060",

        "contype": "4",

        "conaffinity": "2",

        "condim": "3",

        "friction":
            "0.8 0.005 0.0001",

        "rgba":
            "0.2 0.4 0.9 0.7"
    }

)


# ============================================================
# 16. 添加 FT300 force / torque sensor
# ============================================================
#
# MuJoCo：
#
# <force>
#
# 输出：
#
# Fx Fy Fz
#
#
# <torque>
#
# 输出：
#
# Mx My Mz
#
#
# 合起来就是：
#
# W =
#
# [Fx Fy Fz Mx My Mz]
# ============================================================

sensor_xml = root.find(
    "sensor"
)


if sensor_xml is None:

    sensor_xml = ET.SubElement(
        root,
        "sensor"
    )


ET.SubElement(

    sensor_xml,

    "force",

    {
        "name": "ft300_force",

        "site": "ft300_site"
    }

)


ET.SubElement(

    sensor_xml,

    "torque",

    {
        "name": "ft300_torque",

        "site": "ft300_site"
    }

)


# ============================================================
# 17. 保存生成后的模型
# ============================================================

tree.write(

    ft300_xml_path,

    encoding="utf-8",

    xml_declaration=True

)


# ============================================================
# 18. 加载带 FT300 的 MuJoCo 模型
# ============================================================

mj_model = mujoco.MjModel.from_xml_path(
    str(ft300_xml_path)
)

mj_data = mujoco.MjData(
    mj_model
)


mj_data.qpos[:] = q_initial_mj
mj_data.qvel[:] = 0.0


mujoco.mj_forward(
    mj_model,
    mj_data
)


# ============================================================
# 19. 获取 sensor ID
# ============================================================

force_sensor_id = mujoco.mj_name2id(

    mj_model,

    mujoco.mjtObj.mjOBJ_SENSOR,

    "ft300_force"

)


torque_sensor_id = mujoco.mj_name2id(

    mj_model,

    mujoco.mjtObj.mjOBJ_SENSOR,

    "ft300_torque"

)


ft300_site_id = mujoco.mj_name2id(

    mj_model,

    mujoco.mjtObj.mjOBJ_SITE,

    "ft300_site"

)


probe_geom_id = mujoco.mj_name2id(

    mj_model,

    mujoco.mjtObj.mjOBJ_GEOM,

    "tcp_probe"

)


wall_geom_id = mujoco.mj_name2id(

    mj_model,

    mujoco.mjtObj.mjOBJ_GEOM,

    "contact_wall"

)


# ============================================================
# 20. Sensor 在 sensordata 中的位置
# ============================================================

force_sensor_adr = (
    mj_model.sensor_adr[
        force_sensor_id
    ]
)


torque_sensor_adr = (
    mj_model.sensor_adr[
        torque_sensor_id
    ]
)


# ============================================================
# 21. 阻抗平衡点运动
# ============================================================
#
# 和上一课保持一样：
#
# 最终 equilibrium：
#
# +45 mm
#
#
# 墙实际挡住 TCP：
#
# 大约 +35 mm
#
#
# 所以：
#
# spring compression
#
# ≈ 10 mm
#
#
# Kx = 200 N/m
#
# 理论力：
#
# ≈ 2 N
# ============================================================

delta_equilibrium = np.array([
    0.045,
    0.0,
    0.0
])


motion_start_time = 1.0
motion_duration = 4.0


# ============================================================
# 22. Smoothstep
# ============================================================

def smoothstep(t):

    if t <= motion_start_time:

        return 0.0, 0.0


    if t >= (
        motion_start_time
        +
        motion_duration
    ):

        return 1.0, 0.0


    r = (

        (t - motion_start_time)

        /

        motion_duration

    )


    s = (
        3.0 * r**2
        -
        2.0 * r**3
    )


    s_dot = (

        (6.0 * r - 6.0 * r**2)

        /

        motion_duration

    )


    return s, s_dot


# ============================================================
# 23. Cartesian stiffness
# ============================================================

K_position = np.array([
    200.0,
    200.0,
    200.0
])


K_orientation = np.array([
    1.0,
    1.0,
    1.0
])


# ============================================================
# 24. 根据 Lambda 计算阻尼
# ============================================================

J_initial = pin.computeFrameJacobian(

    pin_model,

    pin_data,

    q_initial_pin,

    ee_id,

    pin.LOCAL_WORLD_ALIGNED

)


M_initial = pin.crba(

    pin_model,

    pin_data,

    q_initial_pin

)


M_initial = (
    M_initial
    +
    M_initial.T
) / 2.0


M_inv_JT = np.linalg.solve(

    M_initial,

    J_initial.T

)


Lambda_inv = (

    J_initial

    @

    M_inv_JT

)


Lambda = np.linalg.inv(

    Lambda_inv

    +

    1e-8
    *
    np.eye(6)

)


K_6d = np.concatenate([
    K_position,
    K_orientation
])


D_6d = (

    2.0

    *

    np.sqrt(

        np.maximum(
            np.diag(Lambda),
            1e-8
        )

        *

        K_6d

    )

)


D_position = D_6d[:3]
D_orientation = D_6d[3:]


# ============================================================
# 25. Torque limit
# ============================================================

tau_limit = np.array([
    50.0,
    50.0,
    28.0,
    12.0,
    12.0,
    3.0
])


# ============================================================
# 26. 6D Cartesian impedance controller
# ============================================================

def controller():

    q_mj = mj_data.qpos.copy()
    dq = mj_data.qvel.copy()


    q_pin = mujoco_to_pinocchio(
        q_mj
    )


    # --------------------------------------------------------
    # 当前 TCP pose
    # --------------------------------------------------------

    pin.forwardKinematics(
        pin_model,
        pin_data,
        q_pin
    )

    pin.updateFramePlacements(
        pin_model,
        pin_data
    )


    T = pin_data.oMf[ee_id]

    p = T.translation.copy()
    R = T.rotation.copy()


    # --------------------------------------------------------
    # 移动 equilibrium
    # --------------------------------------------------------

    s, s_dot = smoothstep(
        mj_data.time
    )


    p_des = (

        p_tool_initial

        +

        s
        *
        delta_equilibrium

    )


    v_des = (

        s_dot

        *

        delta_equilibrium

    )


    # 姿态保持初始姿态

    R_des = R_tool_initial

    omega_des = np.zeros(3)


    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    e_position = (

        p_des

        -

        p

    )


    # --------------------------------------------------------
    # Orientation error
    # --------------------------------------------------------

    R_error = (

        R_des

        @

        R.T

    )


    e_orientation = pin.log3(
        R_error
    )


    # --------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------

    J = pin.computeFrameJacobian(

        pin_model,

        pin_data,

        q_pin,

        ee_id,

        pin.LOCAL_WORLD_ALIGNED

    )


    twist = (
        J
        @
        dq
    )


    v = twist[:3]

    omega = twist[3:]


    # --------------------------------------------------------
    # Cartesian impedance
    # --------------------------------------------------------

    force_control = (

        K_position
        *
        e_position

        +

        D_position
        *
        (
            v_des
            -
            v
        )

    )


    moment_control = (

        K_orientation
        *
        e_orientation

        +

        D_orientation
        *
        (
            omega_des
            -
            omega
        )

    )


    wrench = np.concatenate([
        force_control,
        moment_control
    ])


    tau_task = (
        J.T
        @
        wrench
    )


    gravity = pin.computeGeneralizedGravity(

        pin_model,

        pin_data,

        q_pin

    )


    tau = (

        gravity

        +

        tau_task

    )


    tau = np.clip(

        tau,

        -tau_limit,

        tau_limit

    )


    mj_data.ctrl[:] = tau


    return (
        e_position,
        e_orientation,
        v
    )


# ============================================================
# 27. 判断 probe 是否接触 wall
# ============================================================

def is_probe_in_contact():

    for contact_id in range(
        mj_data.ncon
    ):

        contact = mj_data.contact[
            contact_id
        ]


        pair = {

            contact.geom1,

            contact.geom2

        }


        if pair == {
            probe_geom_id,
            wall_geom_id
        }:

            return True


    return False


# ============================================================
# 28. 读取 FT300
# ============================================================

def read_ft300():

    # --------------------------------------------------------
    # Force
    #
    # 单位：
    #
    # N
    #
    # 坐标系：
    #
    # ft300_site LOCAL frame
    # --------------------------------------------------------

    force_local = (

        mj_data.sensordata[
            force_sensor_adr
            :
            force_sensor_adr + 3
        ]

        .copy()

    )


    # --------------------------------------------------------
    # Torque
    #
    # 单位：
    #
    # N*m
    #
    # 同样在 FT300 LOCAL frame
    # --------------------------------------------------------

    torque_local = (

        mj_data.sensordata[
            torque_sensor_adr
            :
            torque_sensor_adr + 3
        ]

        .copy()

    )


    # --------------------------------------------------------
    # sensor frame -> WORLD
    # --------------------------------------------------------
    #
    # site_xmat：
    #
    # sensor local
    #
    #       ↓
    #
    # WORLD
    # --------------------------------------------------------

    R_sensor_world = (

        mj_data
        .site_xmat[ft300_site_id]

        .reshape(
            3,
            3
        )

    )


    force_world = (

        R_sensor_world

        @

        force_local

    )


    torque_world = (

        R_sensor_world

        @

        torque_local

    )


    return (
        force_local,
        torque_local,
        force_world,
        torque_world
    )


# ============================================================
# 29. Simulation
# ============================================================

step = 0


with mujoco.viewer.launch_passive(

    mj_model,

    mj_data

) as viewer:


    while viewer.is_running():

        (
            e_position,
            e_orientation,
            v
        ) = controller()


        # ====================================================
        # MuJoCo physics
        # ====================================================

        mujoco.mj_step(

            mj_model,

            mj_data

        )


        # ====================================================
        # 读取 FT300
        # ====================================================

        (
            force_local,
            torque_local,
            force_world,
            torque_world
        ) = read_ft300()


        contact = (
            is_probe_in_contact()
        )


        viewer.sync()


        # ====================================================
        # 30. 精简打印
        # ============================================================
        #
        # 每500步约1秒。
        #
        # 只打印这一课真正需要看的数据。
        # ====================================================

        if step % 500 == 0:

            contact_text = (

                "YES 是"

                if contact

                else

                "NO 否"

            )


            print(
                "\n"
                "========== FT300 Sensor FT300传感器 =========="
            )


            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )


            print(
                f"Contact 接触: "
                f"{contact_text}"
            )


            print(
                f"X compression X方向压缩量: "
                f"{e_position[0] * 1000.0:.3f} mm"
            )


            print(
                "FT300 Force 传感器力 "
                "[Fx Fy Fz] N:"
            )

            print(
                np.round(
                    force_local,
                    3
                )
            )


            print(
                "FT300 Torque 传感器力矩 "
                "[Mx My Mz] N·m:"
            )

            print(
                np.round(
                    torque_local,
                    4
                )
            )


            print(
                f"FT300 World Fx 世界系X力: "
                f"{force_world[0]:.3f} N"
            )


            print(
                f"Orientation error 姿态误差: "
                f"{np.rad2deg(np.linalg.norm(e_orientation)):.3f} deg"
            )


            print(
                f"TCP X velocity TCP X速度: "
                f"{v[0]:.6f} m/s"
            )


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )