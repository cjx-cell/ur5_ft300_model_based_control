#!/usr/bin/env python3
"""
81_ur3_tsid_se3_control.py

81 - TSID 6D end-effector pose control
TSID 六维末端位姿控制

目标：
1. 使用 TaskSE3Equality 控制 robotiq_ft_frame_id 的 6D 位姿；
2. 使用 InverseDynamicsFormulationAccForce + HQP 求解；
3. 读取 TSID 输出 ddq 与 tau；
4. 用 Pinocchio integrate() 做内部动力学积分；
5. 独立检查末端位置/姿态误差与逆动力学残差。

本节仍然先不接 MuJoCo。
原因：我们先把 TSID 的 6D task 本身验证清楚，
下一节再加入 posture / bounds / 多任务层级。
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
# 4. Arm tangent-space indices
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
# 5. Math helpers
# ============================================================

def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


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
# 6. Initial configuration
#
# 不从 neutral 的全零机械臂位形直接开始，
# 因为六维 Jacobian 可能接近奇异。
#
# 用 Pinocchio integrate() 从 neutral 构造一个
# 明确的非奇异初始姿态。
# ============================================================

q_neutral = pin.neutral(
    model
)

initial_delta = np.zeros(
    nv
)

initial_arm_deg = np.array([
    0.0,
    -60.0,
    90.0,
    -30.0,
    45.0,
    0.0,
])

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
# 7. Initial and desired SE3 pose
# ============================================================

H_initial = frame_pose(
    q
)

H_ref = pin.SE3(
    rotation_z(
        np.deg2rad(
            8.0
        )
    )
    @
    H_initial.rotation,

    H_initial.translation
    +
    np.array([
        0.020,
        -0.015,
        0.010,
    ]),
)


print(
    "\n"
    "========== 81 TSID SE3 Control "
    "81 TSID六维末端位姿控制 =========="
)

print(
    f"Robot dimensions 机器人维度: "
    f"nq={nq}, nv={nv}, na={na}"
)

print(
    f"Controlled frame 控制坐标系: "
    f"{FRAME_NAME}"
)

print(
    "Initial arm posture "
    "初始机械臂姿态(deg): "
    f"{initial_arm_deg.tolist()}"
)

print(
    "Target translation change "
    "目标位置变化(mm): "
    "[+20, -15, +10]"
)

print(
    "Target world-Z rotation "
    "目标世界Z轴旋转: +8 deg"
)

print(
    f"Initial Jacobian condition "
    f"初始雅可比条件数: "
    f"{frame_jacobian_condition(q):.2f}"
)


# ============================================================
# 8. TSID formulation
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
# 9. 6D SE3 task
# ============================================================

se3_task = tsid.TaskSE3Equality(
    "task-se3",
    robot,
    FRAME_NAME,
)

# World-aligned error / Jacobian representation.
#
# If this API were unavailable in an older TSID version,
# the exception below would make that explicit.
if hasattr(
    se3_task,
    "useLocalFrame",
):
    se3_task.useLocalFrame(
        False
    )


kp_value = 100.0

se3_task.setKp(
    kp_value
    *
    np.ones(
        6
    )
)

se3_task.setKd(
    2.0
    *
    np.sqrt(
        kp_value
    )
    *
    np.ones(
        6
    )
)

se3_task.setReference(
    H_ref
)


TASK_WEIGHT = 1.0
TASK_PRIORITY = 1

formulation.addMotionTask(
    se3_task,
    TASK_WEIGHT,
    TASK_PRIORITY,
    0.0,
)


# ============================================================
# 10. HQP solver
# ============================================================

solver = tsid.SolverHQuadProgFast(
    "qp-solver"
)

solver.resize(
    formulation.nVar,
    formulation.nEq,
    formulation.nIn,
)


print(
    f"HQP dimensions HQP维度: "
    f"nVar={formulation.nVar}, "
    f"nEq={formulation.nEq}, "
    f"nIn={formulation.nIn}"
)


# ============================================================
# 11. Internal simulation parameters
# ============================================================

dt = 0.001

duration = 5.0

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
# 12. Statistics
# ============================================================

max_dynamic_residual = 0.0
max_tau = 0.0
max_ddq = 0.0

max_position_error = 0.0
max_orientation_error = 0.0

solver_failed = False


# ============================================================
# 13. TSID control loop
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
    # A. Build HQP
    # --------------------------------------------------------

    hqp_data = formulation.computeProblemData(
        t,
        q,
        v,
    )


    # --------------------------------------------------------
    # B. Solve
    # --------------------------------------------------------

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
    # C. TSID outputs
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
    # D. Independent inverse-dynamics check
    # --------------------------------------------------------

    rnea_data = model.createData()

    tau_rnea = pin.rnea(
        model,
        rnea_data,
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
    # E. Independent 6D pose error
    #
    # Position:
    #
    #     e_p = p_ref - p
    #
    # Orientation in world coordinates:
    #
    #     e_R = log3(R_ref R^T)
    # --------------------------------------------------------

    H_current = frame_pose(
        q
    )

    position_error_vector = (
        H_ref.translation
        -
        H_current.translation
    )

    orientation_error_vector = pin.log3(
        H_ref.rotation
        @
        H_current.rotation.T
    )

    position_error = float(
        np.linalg.norm(
            position_error_vector
        )
    )

    orientation_error_deg = float(
        np.rad2deg(
            np.linalg.norm(
                orientation_error_vector
            )
        )
    )


    # --------------------------------------------------------
    # F. Diagnostics
    # --------------------------------------------------------

    jacobian_condition = (
        frame_jacobian_condition(
            q
        )
    )

    arm_speed = float(
        np.linalg.norm(
            v[
                arm_v_indices
            ]
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

    max_position_error = max(
        max_position_error,
        position_error,
    )

    max_orientation_error = max(
        max_orientation_error,
        orientation_error_deg,
    )


    # --------------------------------------------------------
    # G. Compact output every 1 second
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
            "---------- TSID SE3 State "
            "TSID六维状态 ----------"
        )

        print(
            f"Time 时间: "
            f"{t:.2f} s"
        )

        print(
            f"Position error "
            f"位置误差: "
            f"{position_error * 1000.0:.4f} mm"
        )

        print(
            f"Orientation error "
            f"姿态误差: "
            f"{orientation_error_deg:.4f} deg"
        )

        print(
            f"Arm speed "
            f"机械臂速度范数: "
            f"{arm_speed:.5f} rad/s"
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
    # H. Integrate internal state
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
# 14. Final result
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

final_orientation_error_vector = pin.log3(
    H_ref.rotation
    @
    H_final.rotation.T
)

final_orientation_error_deg = float(
    np.rad2deg(
        np.linalg.norm(
            final_orientation_error_vector
        )
    )
)

final_arm_speed = float(
    np.linalg.norm(
        v[
            arm_v_indices
        ]
    )
)


print(
    "\n"
    "========== 81 Result "
    "81结果 =========="
)

print(
    f"Solver status "
    f"求解状态: "
    f"{'FAILED 失败' if solver_failed else 'OPTIMAL 正常'}"
)

print(
    f"Final position error "
    f"最终位置误差: "
    f"{final_position_error * 1000.0:.6f} mm"
)

print(
    f"Final orientation error "
    f"最终姿态误差: "
    f"{final_orientation_error_deg:.6f} deg"
)

print(
    f"Final arm speed "
    f"最终机械臂速度范数: "
    f"{final_arm_speed:.8f} rad/s"
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
