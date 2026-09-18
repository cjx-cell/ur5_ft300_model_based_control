#!/usr/bin/env python3
"""
82_ur3_tsid_multitask_bounds.py

82 - TSID 多任务 + 速度约束

本节目的：
1. 用 TaskSE3Equality 做“主任务”；
2. 用 TaskJointPosture 做“次要姿态目标”；
3. 用 TaskJointBounds 加入硬速度约束；
4. 观察 TSID 如何在“主任务、冗余姿态、硬约束”之间分配自由度。

为了让 posture task 真正能发挥作用，本节故意把 SE3 task 的 mask
设成“只控制末端 XYZ 位置”，不锁死末端姿态。

也就是说：
    主任务：TCP 位置到达目标；
    次任务：在不明显破坏主任务的前提下，让关节姿态靠近期望；
    硬约束：所有关节速度不能超过设定上限。

这正是 TSID 比“直接 J^{-1} 求 qdd”更有价值的地方。
"""

from pathlib import Path
import warnings

import numpy as np
import pinocchio as pin
import tsid


# ============================================================
# 0. Ignore harmless duplicate Boost.Python converter warning
# ============================================================

warnings.filterwarnings(
    "ignore",
    message=".*to-Python converter.*already registered.*",
)


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

URDF_PATH = (
    SCRIPT_DIR
    / "ur3_ft300_robotiq_force_control.urdf"
)

if not URDF_PATH.exists():
    raise FileNotFoundError(
        f"找不到 URDF:\n{URDF_PATH}"
    )


# ============================================================
# 2. Load TSID robot
# ============================================================

package_dirs = pin.StdVec_StdString()
package_dirs.append(
    str(
        URDF_PATH.parent
    )
)

robot = tsid.RobotWrapper(
    str(
        URDF_PATH
    ),
    package_dirs,
    False,
)

model = robot.model()

nq = int(model.nq)
nv = int(model.nv)
na = int(robot.na)


# ============================================================
# 3. Controlled frame
# ============================================================

FRAME_NAME = "robotiq_ft_frame_id"

frame_id = model.getFrameId(
    FRAME_NAME
)

if frame_id >= model.nframes:
    raise RuntimeError(
        f"找不到 frame: {FRAME_NAME}"
    )


# ============================================================
# 4. Arm indices
# ============================================================

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

arm_v_indices = []

for name in ARM_JOINT_NAMES:

    joint_id = model.getJointId(
        name
    )

    if joint_id == 0:
        raise RuntimeError(
            f"找不到机械臂关节: {name}"
        )

    joint = model.joints[
        joint_id
    ]

    if joint.nv != 1:
        raise RuntimeError(
            f"{name} 的 nv={joint.nv}，"
            "本实验期望单自由度关节"
        )

    arm_v_indices.append(
        int(
            joint.idx_v
        )
    )

arm_v_indices = np.asarray(
    arm_v_indices,
    dtype=int,
)


# ============================================================
# 5. Helpers
# ============================================================

def frame_pose(
    q_configuration,
):
    data = model.createData()

    pin.forwardKinematics(
        model,
        data,
        q_configuration,
    )

    pin.updateFramePlacements(
        model,
        data,
    )

    placement = data.oMf[
        frame_id
    ]

    return pin.SE3(
        placement.rotation.copy(),
        placement.translation.copy(),
    )


def frame_jacobian_condition(
    q_configuration,
):
    data = model.createData()

    pin.computeJointJacobians(
        model,
        data,
        q_configuration,
    )

    pin.updateFramePlacements(
        model,
        data,
    )

    J = pin.getFrameJacobian(
        model,
        data,
        frame_id,
        pin.LOCAL_WORLD_ALIGNED,
    )

    J_arm = J[
        :,
        arm_v_indices
    ]

    singular_values = np.linalg.svd(
        J_arm,
        compute_uv=False,
    )

    if singular_values[-1] < 1e-12:
        return np.inf

    return float(
        singular_values[0]
        /
        singular_values[-1]
    )


# ============================================================
# 6. Initial state
# ============================================================

q_neutral = pin.neutral(
    model
)

initial_arm_deg = np.array([
    0.0,
    -60.0,
    90.0,
    -30.0,
    45.0,
    0.0,
])

initial_delta = np.zeros(
    nv
)

initial_delta[
    arm_v_indices
] = np.deg2rad(
    initial_arm_deg
)

q = pin.integrate(
    model,
    q_neutral,
    initial_delta,
)

v = np.zeros(
    nv
)


# ============================================================
# 7. Primary task target: TCP position
#
# 这里用 TaskSE3Equality，但 mask 只打开 XYZ 三个位置自由度。
#
# mask:
#
#     [1, 1, 1, 0, 0, 0]
#
# 前三维：线位置任务
# 后三维：姿态不作为主任务约束
#
# 这样机械臂还保留姿态冗余，可由 posture task 使用。
# ============================================================

H_initial = frame_pose(
    q
)

H_ref = pin.SE3(
    H_initial.rotation.copy(),
    H_initial.translation
    +
    np.array([
        0.040,
        -0.020,
        0.020,
    ]),
)


# ============================================================
# 8. Secondary posture target
#
# 这个目标故意改变多个机械臂关节。
# 由于主任务只约束 TCP 位置，TSID 可以利用剩余自由度，
# 在保持 TCP 位置的同时尽量靠近这个姿态目标。
# ============================================================

posture_delta = np.zeros(
    nv
)

posture_change_deg = np.array([
    +15.0,
    -10.0,
    +10.0,
    +20.0,
    -15.0,
    +20.0,
])

posture_delta[
    arm_v_indices
] = np.deg2rad(
    posture_change_deg
)

q_posture_ref = pin.integrate(
    model,
    q,
    posture_delta,
)


# ============================================================
# 9. TSID formulation
# ============================================================

formulation = (
    tsid.InverseDynamicsFormulationAccForce(
        "tsid",
        robot,
        False,
    )
)

formulation.computeProblemData(
    0.0,
    q,
    v,
)


# ============================================================
# 10. Primary TaskSE3Equality
# ============================================================

se3_task = tsid.TaskSE3Equality(
    "task-se3-position",
    robot,
    FRAME_NAME,
)

if hasattr(
    se3_task,
    "useLocalFrame",
):
    se3_task.useLocalFrame(
        False
    )

se3_task.setMask(
    np.array([
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ])
)

kp_se3 = 64.0

se3_task.setKp(
    kp_se3
    *
    np.ones(
        3
    )
)

se3_task.setKd(
    2.0
    *
    np.sqrt(
        kp_se3
    )
    *
    np.ones(
        3
    )
)


# TSID 1.10 Python binding expects TrajectorySample.
se3_reference_sample = tsid.TrajectorySample(
    12,
    6,
)

se3_reference_value = np.asarray(
    tsid.SE3ToVector(
        H_ref
    )
).reshape(
    -1
)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        DeprecationWarning,
    )

    se3_reference_sample.pos(
        se3_reference_value
    )

    se3_reference_sample.vel(
        np.zeros(
            6
        )
    )

    se3_reference_sample.acc(
        np.zeros(
            6
        )
    )

se3_task.setReference(
    se3_reference_sample
)


# ============================================================
# 11. Secondary posture task
# ============================================================

posture_task = tsid.TaskJointPosture(
    "task-posture",
    robot,
)

kp_posture = 16.0

posture_task.setKp(
    kp_posture
    *
    np.ones(
        na
    )
)

posture_task.setKd(
    2.0
    *
    np.sqrt(
        kp_posture
    )
    *
    np.ones(
        na
    )
)


posture_sample = tsid.TrajectorySample(
    nq,
    nv,
)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        DeprecationWarning,
    )

    posture_sample.pos(
        q_posture_ref
    )

    posture_sample.vel(
        np.zeros(
            nv
        )
    )

    posture_sample.acc(
        np.zeros(
            nv
        )
    )

posture_task.setReference(
    posture_sample
)


# ============================================================
# 12. Hard velocity bounds
#
# TaskJointBounds creates inequality constraints.
#
# arm:
#     |dq| <= 0.35 rad/s
#
# other actuated joints:
#     |dq| <= 1.0 rad/s
# ============================================================

dt = 0.001

joint_bounds_task = tsid.TaskJointBounds(
    "task-joint-bounds",
    robot,
    dt,
)

velocity_lower = (
    -1.0
    *
    np.ones(
        na
    )
)

velocity_upper = (
    +1.0
    *
    np.ones(
        na
    )
)

velocity_lower[
    arm_v_indices
] = -0.35

velocity_upper[
    arm_v_indices
] = +0.35

joint_bounds_task.setVelocityBounds(
    velocity_lower,
    velocity_upper,
)


# ============================================================
# 13. Add tasks
#
# priority 0:
#     hard constraint / joint velocity bounds
#
# priority 1:
#     primary Cartesian position + low-weight posture regularization
#
# 这里采用官方常见结构：
#     bounds -> priority 0
#     motion tasks -> priority 1
#
# posture weight 很小，所以主任务优先。
# ============================================================

SE3_WEIGHT = 1.0

POSTURE_WEIGHT = 1.0e-3

formulation.addMotionTask(
    joint_bounds_task,
    1.0,
    0,
    0.0,
)

formulation.addMotionTask(
    se3_task,
    SE3_WEIGHT,
    1,
    0.0,
)

formulation.addMotionTask(
    posture_task,
    POSTURE_WEIGHT,
    1,
    0.0,
)


# ============================================================
# 14. Solver
# ============================================================

solver = tsid.SolverHQuadProgFast(
    "qp-solver"
)

solver.resize(
    formulation.nVar,
    formulation.nEq,
    formulation.nIn,
)


# ============================================================
# 15. Header
# ============================================================

print(
    "\n"
    "========== 82 TSID Multi-task + Bounds "
    "82 TSID多任务与约束 =========="
)

print(
    f"Robot dimensions 机器人维度: "
    f"nq={nq}, nv={nv}, na={na}"
)

print(
    "Primary task 主任务: "
    "TCP XYZ position TCP三维位置"
)

print(
    "TCP target change TCP目标变化(mm): "
    "[+40, -20, +20]"
)

print(
    "Secondary task 次任务: "
    "joint posture 关节姿态"
)

print(
    "Posture target change "
    "姿态目标关节变化(deg): "
    f"{posture_change_deg.tolist()}"
)

print(
    "Arm speed limit "
    "机械臂速度限制: "
    "±0.35 rad/s"
)

print(
    f"HQP dimensions HQP维度: "
    f"nVar={formulation.nVar}, "
    f"nEq={formulation.nEq}, "
    f"nIn={formulation.nIn}"
)

print(
    f"Initial Jacobian condition "
    f"初始雅可比条件数: "
    f"{frame_jacobian_condition(q):.2f}"
)


# ============================================================
# 16. Simulation parameters
# ============================================================

duration = 6.0

number_of_steps = int(
    duration
    /
    dt
)

print_every_steps = int(
    1.0
    /
    dt
)


# ============================================================
# 17. Statistics
# ============================================================

max_dynamic_residual = 0.0
max_tau = 0.0
max_ddq = 0.0

max_arm_speed = 0.0

solver_failed = False


# ============================================================
# 18. Loop
# ============================================================

for step in range(
    number_of_steps + 1
):

    t = (
        step
        *
        dt
    )


    # --------------------------------------------------------
    # A. HQP
    # --------------------------------------------------------

    hqp_data = formulation.computeProblemData(
        t,
        q,
        v,
    )

    solution = solver.solve(
        hqp_data
    )

    if solution.status != 0:

        print(
            f"\nQP solve failed "
            f"QP求解失败: "
            f"t={t:.3f}s, "
            f"status={solution.status}"
        )

        solver_failed = True

        break


    # --------------------------------------------------------
    # B. Outputs
    # --------------------------------------------------------

    tau = np.asarray(
        formulation.getActuatorForces(
            solution
        )
    ).reshape(
        -1
    )

    ddq = np.asarray(
        formulation.getAccelerations(
            solution
        )
    ).reshape(
        -1
    )


    # --------------------------------------------------------
    # C. Dynamics check
    # --------------------------------------------------------

    tau_rnea = pin.rnea(
        model,
        model.createData(),
        q,
        v,
        ddq,
    )

    dynamic_residual = float(
        np.linalg.norm(
            tau
            -
            tau_rnea
        )
    )


    # --------------------------------------------------------
    # D. Primary task error
    # --------------------------------------------------------

    H_current = frame_pose(
        q
    )

    position_error_vector = (
        H_ref.translation
        -
        H_current.translation
    )

    position_error = float(
        np.linalg.norm(
            position_error_vector
        )
    )


    # --------------------------------------------------------
    # E. Orientation is free in the primary task.
    #
    # We print how much it changed from the initial orientation.
    # --------------------------------------------------------

    orientation_change_vector = pin.log3(
        H_current.rotation
        @
        H_initial.rotation.T
    )

    orientation_change_deg = float(
        np.rad2deg(
            np.linalg.norm(
                orientation_change_vector
            )
        )
    )


    # --------------------------------------------------------
    # F. Secondary posture error
    # --------------------------------------------------------

    posture_error = pin.difference(
        model,
        q,
        q_posture_ref,
    )

    arm_posture_error_deg = float(
        np.linalg.norm(
            np.rad2deg(
                posture_error[
                    arm_v_indices
                ]
            )
        )
    )


    # --------------------------------------------------------
    # G. Bounds diagnostics
    # --------------------------------------------------------

    arm_speed_max_now = float(
        np.max(
            np.abs(
                v[
                    arm_v_indices
                ]
            )
        )
    )

    arm_speed_ratio = (
        arm_speed_max_now
        /
        0.35
    )

    jacobian_condition = (
        frame_jacobian_condition(
            q
        )
    )


    max_dynamic_residual = max(
        max_dynamic_residual,
        dynamic_residual,
    )

    max_tau = max(
        max_tau,
        float(
            np.max(
                np.abs(
                    tau
                )
            )
        ),
    )

    max_ddq = max(
        max_ddq,
        float(
            np.max(
                np.abs(
                    ddq
                )
            )
        ),
    )

    max_arm_speed = max(
        max_arm_speed,
        arm_speed_max_now,
    )


    # --------------------------------------------------------
    # H. Compact print every second
    # --------------------------------------------------------

    if (
        step
        %
        print_every_steps
        ==
        0
    ):

        print(
            "\n"
            "---------- 82 State "
            "82状态 ----------"
        )

        print(
            f"Time 时间: "
            f"{t:.2f} s"
        )

        print(
            f"Primary position error "
            f"主任务位置误差: "
            f"{position_error * 1000.0:.4f} mm"
        )

        print(
            f"Secondary posture error "
            f"次任务机械臂姿态误差: "
            f"{arm_posture_error_deg:.3f} deg"
        )

        print(
            f"Free orientation change "
            f"自由姿态变化: "
            f"{orientation_change_deg:.3f} deg"
        )

        print(
            f"Arm max speed "
            f"机械臂当前最大速度: "
            f"{arm_speed_max_now:.4f} rad/s"
        )

        print(
            f"Speed-limit usage "
            f"速度约束使用率: "
            f"{100.0 * arm_speed_ratio:.1f}%"
        )

        print(
            f"Jacobian condition "
            f"雅可比条件数: "
            f"{jacobian_condition:.2f}"
        )

        print(
            f"Max torque "
            f"当前最大关节力矩: "
            f"{np.max(np.abs(tau)):.3f} Nm"
        )

        print(
            f"Dynamics residual "
            f"动力学残差: "
            f"{dynamic_residual:.3e}"
        )


    # --------------------------------------------------------
    # I. Integrate
    # --------------------------------------------------------

    if step < number_of_steps:

        v_mid = (
            v
            +
            0.5
            *
            dt
            *
            ddq
        )

        q = pin.integrate(
            model,
            q,
            dt
            *
            v_mid,
        )

        v = (
            v
            +
            dt
            *
            ddq
        )


# ============================================================
# 19. Final result
# ============================================================

H_final = frame_pose(
    q
)

final_position_error = float(
    np.linalg.norm(
        H_ref.translation
        -
        H_final.translation
    )
)

final_orientation_change = float(
    np.rad2deg(
        np.linalg.norm(
            pin.log3(
                H_final.rotation
                @
                H_initial.rotation.T
            )
        )
    )
)

final_posture_error = pin.difference(
    model,
    q,
    q_posture_ref,
)

final_arm_posture_error_deg = float(
    np.linalg.norm(
        np.rad2deg(
            final_posture_error[
                arm_v_indices
            ]
        )
    )
)


print(
    "\n"
    "========== 82 Result "
    "82结果 =========="
)

print(
    f"Solver status "
    f"求解状态: "
    f"{'FAILED 失败' if solver_failed else 'OPTIMAL 正常'}"
)

print(
    f"Final primary position error "
    f"最终主任务位置误差: "
    f"{final_position_error * 1000.0:.6f} mm"
)

print(
    f"Final secondary posture error "
    f"最终次任务机械臂姿态误差: "
    f"{final_arm_posture_error_deg:.4f} deg"
)

print(
    f"Final free orientation change "
    f"最终自由姿态变化: "
    f"{final_orientation_change:.4f} deg"
)

print(
    f"Max arm speed "
    f"机械臂最大速度: "
    f"{max_arm_speed:.6f} rad/s"
)

print(
    f"Configured arm speed limit "
    f"机械臂速度上限: "
    f"0.350000 rad/s"
)

print(
    f"Maximum speed-limit usage "
    f"最大速度约束使用率: "
    f"{100.0 * max_arm_speed / 0.35:.2f}%"
)

print(
    f"Final Jacobian condition "
    f"最终雅可比条件数: "
    f"{frame_jacobian_condition(q):.2f}"
)

print(
    f"Max dynamics residual "
    f"最大动力学残差: "
    f"{max_dynamic_residual:.3e}"
)

print(
    f"Max torque "
    f"最大关节力矩: "
    f"{max_tau:.3f} Nm"
)

print(
    f"Max acceleration "
    f"最大关节加速度: "
    f"{max_ddq:.3f} rad/s^2"
)
