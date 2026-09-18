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

work_dir = Path(__file__).parent

base_xml_path = work_dir / "ur3_converted.xml"

# 这个脚本会自动从 ur3_converted.xml 生成一个新的接触场景。
#
# 不修改已经验证好的 ur3_converted.xml。

contact_xml_path = work_dir / "ur3_contact_scene.xml"


# ============================================================
# 2. Pinocchio 模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

pin_model = pin.buildModelFromUrdf(
    urdf_path
)

pin_data = pin_model.createData()


ee_name = "tool0"

ee_id = pin_model.getFrameId(
    ee_name
)

print(
    "Pinocchio tool0 frame id =",
    ee_id
)


# ============================================================
# 3. MuJoCo q -> Pinocchio q
# ============================================================

def mujoco_to_pinocchio(q_mj):

    q_pin = pin.neutral(
        pin_model
    )

    # 前5个关节直接对应
    q_pin[:5] = q_mj[:5]

    # wrist3 是 JointModelRUBZ
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
# 目的：
#
# 我们首先要知道 wrist_3_link 的 WORLD pose，
# 再结合 Pinocchio 的 tool0 WORLD position，
# 计算：
#
# tool0 在 wrist_3_link body frame 中的局部坐标。
#
# 然后才能准确把一个 sphere geom 放到 TCP。
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
# 6. Pinocchio 得到 tool0 WORLD position
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


T_tool_initial = pin_data.oMf[ee_id]

p_tool_initial = (
    T_tool_initial.translation.copy()
)

R_tool_initial = (
    T_tool_initial.rotation.copy()
)


print(
    "\nInitial tool0 WORLD position:"
)

print(
    p_tool_initial
)


# ============================================================
# 7. 找 MuJoCo wrist_3_link
# ============================================================

wrist_body_id = mujoco.mj_name2id(
    base_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "wrist_3_link"
)


if wrist_body_id == -1:

    raise RuntimeError(
        "MuJoCo 模型中找不到 wrist_3_link"
    )


# wrist_3_link body 原点的 WORLD position

p_wrist_world = (
    base_data.xpos[wrist_body_id].copy()
)


# body -> WORLD rotation
#
# xmat 是9元素，
# reshape 成3x3。

R_wrist_world = (
    base_data.xmat[wrist_body_id]
    .reshape(3, 3)
    .copy()
)


# ============================================================
# 8. 计算 tool0 在 wrist_3_link 中的 local position
# ============================================================
#
# WORLD：
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


print(
    "\ntool0 position in wrist_3_link frame:"
)

print(
    p_tool_local
)


# ============================================================
# 9. 自动生成接触场景 XML
# ============================================================

tree = ET.parse(
    base_xml_path
)

root = tree.getroot()


# ------------------------------------------------------------
# 找 wrist_3_link XML body
# ------------------------------------------------------------

wrist_body_xml = root.find(
    ".//body[@name='wrist_3_link']"
)


if wrist_body_xml is None:

    raise RuntimeError(
        "XML 中找不到 wrist_3_link body"
    )


# ------------------------------------------------------------
# 如果之前生成过 probe，先删除
# ------------------------------------------------------------

for child in list(wrist_body_xml):

    if (
        child.tag == "geom"
        and
        child.attrib.get("name") == "tcp_probe"
    ):

        wrist_body_xml.remove(
            child
        )


# ============================================================
# 10. 给 TCP 添加 collision sphere
# ============================================================
#
# 这个 sphere：
#
# - 中心位于 tool0
# - 半径 15 mm
# - mass = 0，不改变机器人物理质量
#
#
# Collision bit：
#
# contype = 2
# conaffinity = 4
#
# 后面的墙：
#
# contype = 4
# conaffinity = 2
#
#
# 因此：
#
# probe <-> wall
#
# 会发生碰撞。
#
#
# 但默认机器人 mesh 通常是 1 / 1，
#
# 所以不会和新墙发生碰撞。
# ============================================================

ET.SubElement(
    wrist_body_xml,
    "geom",
    {
        "name": "tcp_probe",

        "type": "sphere",

        "pos":
            f"{p_tool_local[0]} "
            f"{p_tool_local[1]} "
            f"{p_tool_local[2]}",

        "size": "0.015",

        "mass": "0",

        "contype": "2",

        "conaffinity": "4",

        "condim": "3",

        "friction": "0.8 0.005 0.0001",

        "rgba": "0.9 0.2 0.2 1"
    }
)


# ============================================================
# 11. 创建墙
# ============================================================

worldbody_xml = root.find(
    "worldbody"
)


if worldbody_xml is None:

    raise RuntimeError(
        "XML 中找不到 worldbody"
    )


# 如果以前生成过墙，先删除

for child in list(worldbody_xml):

    if (
        child.tag == "geom"
        and
        child.attrib.get("name") == "contact_wall"
    ):

        worldbody_xml.remove(
            child
        )


# ------------------------------------------------------------
# 墙位置
# ------------------------------------------------------------
#
# 初始 tool0:
#
# x = x0
#
#
# 墙中心：
#
# x0 + 0.060 m
#
#
# 墙厚度：
#
# 2 * 0.010 = 20 mm
#
#
# 所以墙朝向机器人这一面：
#
# x0 + 0.050 m
#
#
# TCP sphere 半径：
#
# 15 mm
#
#
# 因此理论第一次接触大约发生在：
#
# TCP center:
#
# x0 + 0.035 m
#
# ============================================================

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

        # MuJoCo box size 是 half-size
        #
        # 实际尺寸：
        #
        # X = 20 mm
        # Y = 120 mm
        # Z = 120 mm

        "size": "0.010 0.060 0.060",

        "contype": "4",

        "conaffinity": "2",

        "condim": "3",

        "friction": "0.8 0.005 0.0001",

        "rgba": "0.2 0.4 0.9 0.7"
    }
)


# 保存新的 contact XML

tree.write(
    contact_xml_path,
    encoding="utf-8",
    xml_declaration=True
)


print(
    "\nGenerated contact scene:"
)

print(
    contact_xml_path
)


# ============================================================
# 12. 加载新的 Contact MuJoCo model
# ============================================================

mj_model = mujoco.MjModel.from_xml_path(
    str(contact_xml_path)
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
# 13. 获取 contact geom ID
# ============================================================

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


if probe_geom_id == -1:

    raise RuntimeError(
        "找不到 tcp_probe geom"
    )


if wall_geom_id == -1:

    raise RuntimeError(
        "找不到 contact_wall geom"
    )


print(
    "\ntcp_probe geom id =",
    probe_geom_id
)

print(
    "contact_wall geom id =",
    wall_geom_id
)


# ============================================================
# 14. 阻抗平衡点移动距离
# ============================================================
#
# TCP sphere 理论大约：
#
# +35 mm
#
# 开始碰墙。
#
#
# 我们把最终 equilibrium 设置成：
#
# +45 mm
#
#
# 因此最终：
#
# equilibrium point
#
# 比实际无法继续前进的 TCP
#
# 多约 10 mm。
#
#
# 如果：
#
# Kx = 200 N/m
#
# 那么理论推墙力大约：
#
# F = K * dx
#
#   = 200 * 0.01
#
#   ≈ 2 N
#
#
# 这就把上一课：
#
# 2N -> 10mm
#
# 反过来：
#
# 10mm spring compression -> 2N contact force
# ============================================================

delta_equilibrium = np.array([
    0.045,
    0.0,
    0.0
])


motion_start_time = 1.0
motion_duration = 4.0


# ============================================================
# 15. Smoothstep
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
# 16. Cartesian stiffness
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
# 17. 根据 Lambda 计算阻尼
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


print(
    "\nD_position =",
    D_position
)

print(
    "D_orientation =",
    D_orientation
)


# ============================================================
# 18. Torque limit
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
# 19. 6D Cartesian impedance controller
# ============================================================

def controller():

    # --------------------------------------------------------
    # 当前 joint state
    # --------------------------------------------------------

    q_mj = mj_data.qpos.copy()
    dq = mj_data.qvel.copy()


    q_pin = mujoco_to_pinocchio(
        q_mj
    )


    # --------------------------------------------------------
    # FK
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
    # 移动 impedance equilibrium
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


    # 姿态保持初始值

    R_des = R_tool_initial


    omega_des = np.zeros(3)


    # --------------------------------------------------------
    # 位置误差
    # --------------------------------------------------------

    e_position = (
        p_des
        -
        p
    )


    # --------------------------------------------------------
    # 姿态误差
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
    # Translation impedance
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


    # --------------------------------------------------------
    # Orientation impedance
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 6D wrench
    # --------------------------------------------------------

    wrench = np.concatenate([
        force_control,
        moment_control
    ])


    # --------------------------------------------------------
    # Wrench -> joint torque
    # --------------------------------------------------------

    tau_task = (
        J.T
        @
        wrench
    )


    # --------------------------------------------------------
    # Gravity compensation
    # --------------------------------------------------------

    gravity = pin.computeGeneralizedGravity(
        pin_model,
        pin_data,
        q_pin
    )


    # --------------------------------------------------------
    # Final torque
    # --------------------------------------------------------

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
        p,
        p_des,
        e_position,
        e_orientation,
        v,
        omega,
        force_control,
        moment_control,
        tau,
        s
    )


# ============================================================
# 20. 读取 probe-wall contact wrench
# ============================================================
#
# mj_contactForce():
#
# 返回：
#
# [
#   Fx,
#   Fy,
#   Fz,
#   Mx,
#   My,
#   Mz
# ]
#
# 但是是在 CONTACT FRAME 中表达。
#
#
# mjContact.frame:
#
# 三个 contact frame 轴按 ROW 存储：
#
# frame[0] = contact X / normal
# frame[1] = tangent Y
# frame[2] = tangent Z
#
#
# 因此：
#
# world_force =
#
#     frame.T @ contact_force
# ============================================================

def read_probe_wall_contact():

    total_world_force = np.zeros(3)

    total_world_torque = np.zeros(3)

    contact_count = 0


    for contact_id in range(
        mj_data.ncon
    ):

        contact = mj_data.contact[
            contact_id
        ]


        g1 = contact.geom1
        g2 = contact.geom2


        # 只关心：
        #
        # tcp_probe <-> contact_wall

        correct_pair = (

            (
                g1 == probe_geom_id
                and
                g2 == wall_geom_id
            )

            or

            (
                g1 == wall_geom_id
                and
                g2 == probe_geom_id
            )

        )


        if not correct_pair:
            continue


        contact_count += 1


        # -----------------------------------------------
        # 接触 frame 中的 6D wrench
        # -----------------------------------------------

        wrench_contact = np.zeros(
            6
        )


        mujoco.mj_contactForce(
            mj_model,
            mj_data,
            contact_id,
            wrench_contact
        )


        force_contact = (
            wrench_contact[:3]
        )

        torque_contact = (
            wrench_contact[3:]
        )


        # -----------------------------------------------
        # Contact frame orientation
        #
        # MuJoCo这里的frame axis按行保存。
        # -----------------------------------------------

        contact_frame = (

            np.array(
                contact.frame
            )

            .reshape(
                3,
                3
            )

        )


        # -----------------------------------------------
        # contact -> WORLD
        # -----------------------------------------------

        force_world = (

            contact_frame.T

            @

            force_contact

        )


        torque_world = (

            contact_frame.T

            @

            torque_contact

        )


        total_world_force += (
            force_world
        )

        total_world_torque += (
            torque_world
        )


    return (
        contact_count,
        total_world_force,
        total_world_torque
    )


# ============================================================
# 21. Simulation
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data
) as viewer:


    while viewer.is_running():

        (
            p,
            p_des,
            e_position,
            e_orientation,
            v,
            omega,
            force_control,
            moment_control,
            tau,
            progress
        ) = controller()


        # ----------------------------------------------------
        # MuJoCo physics
        # ----------------------------------------------------

        mujoco.mj_step(
            mj_model,
            mj_data
        )


        # ----------------------------------------------------
        # 接触力必须在 mj_step 后读取
        #
        # 因为这时 constraint/contact force 已经求解。
        # ----------------------------------------------------

        (
            contact_count,
            contact_force_world,
            contact_torque_world
        ) = read_probe_wall_contact()


        viewer.sync()


        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        if step % 250 == 0:

            print(
                "\n"
                "========== REAL CONTACT TEST =========="
            )


            print(
                "time =",
                mj_data.time
            )


            print(
                "trajectory progress =",
                progress
            )


            print(
                "\nTCP current position:"
            )

            print(
                p
            )


            print(
                "\nImpedance equilibrium position:"
            )

            print(
                p_des
            )


            print(
                "\nPosition error:"
            )

            print(
                e_position
            )


            print(
                "X spring compression [mm] =",
                e_position[0]
                *
                1000.0
            )


            print(
                "\nOrientation error [deg] =",
                np.rad2deg(
                    np.linalg.norm(
                        e_orientation
                    )
                )
            )


            print(
                "\nControl Cartesian force:"
            )

            print(
                force_control
            )


            print(
                "\nContact count =",
                contact_count
            )


            print(
                "MuJoCo total contact force WORLD:"
            )

            print(
                contact_force_world
            )


            print(
                "Contact force magnitude [N] =",
                np.linalg.norm(
                    contact_force_world
                )
            )


            print(
                "\nMuJoCo contact torque WORLD:"
            )

            print(
                contact_torque_world
            )


            print(
                "\nTCP linear velocity:"
            )

            print(
                v
            )


            print(
                "\nTCP angular velocity:"
            )

            print(
                omega
            )


            print(
                "\nJoint torque:"
            )

            print(
                tau
            )


            print(
                "\nAll MuJoCo contacts =",
                mj_data.ncon
            )


            print(
                "qfrc_constraint ="
            )

            print(
                mj_data.qfrc_constraint
            )


        step += 1


        # 和之前一样：
        #
        # 不自动 break。
        #
        # 防止你这台机器在 passive viewer
        # 自动退出时再次出现 segmentation fault。
        #
        # 实验完成后手动关闭 viewer。

        time.sleep(
            mj_model.opt.timestep
        )