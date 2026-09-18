#!/usr/bin/env python3
"""
80_ur3_tsid_posture_control.py

80A - TSID 关节姿态控制最小完整实验

目的：
1. 使用 TSID RobotWrapper 读取完整 UR3 + FT300 + Robotiq URDF；
2. 使用 TaskJointPosture 构造关节姿态任务；
3. 使用 InverseDynamicsFormulationAccForce 构造逆动力学 HQP；
4. 使用 SolverHQuadProgFast 求解；
5. 读取 TSID 输出的关节加速度 ddq 和关节力矩 tau；
6. 使用 Pinocchio integrate() 做内部状态积分；
7. 验证 TSID 输出满足逆动力学：
       M(q) ddq + h(q,dq) = tau

说明：
- 80A 先不接 MuJoCo，单独验证 TSID 本身。
- 当前完整 URDF 为 nq=13, nv=12, na=12。
- 由于 MuJoCo 只有 6 个机械臂电机 + 1 个夹爪 actuator，
  而 TSID 将完整 URDF 的 12 个速度自由度视为 actuated，
  直接连接会把“TSID原理”和“执行器映射问题”混在一起。
"""

from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*to-Python converter.*already registered.*",
)

import numpy as np
import pinocchio as pin
import tsid


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "ur3_ft300_robotiq_force_control.urdf"

if not URDF_PATH.exists():
    raise FileNotFoundError(
        f"找不到 URDF:\n{URDF_PATH}"
    )


# ============================================================
# 2. Load TSID robot
# ============================================================

package_dirs = pin.StdVec_StdString()
package_dirs.append(str(URDF_PATH.parent))

robot = tsid.RobotWrapper(
    str(URDF_PATH),
    package_dirs,
    False,
)

model = robot.model()

nq = int(model.nq)
nv = int(model.nv)
na = int(robot.na)

print(
    "\n"
    "========== 80A TSID Posture Control "
    "80A TSID关节姿态控制 =========="
)

print(
    f"Robot dimensions 机器人维度: "
    f"nq={nq}, nv={nv}, na={na}"
)


# ============================================================
# 3. Arm joint indices
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
    joint_id = model.getJointId(name)

    if joint_id == 0:
        raise RuntimeError(
            f"TSID / Pinocchio 找不到关节: {name}"
        )

    joint = model.joints[joint_id]

    if joint.nv != 1:
        raise RuntimeError(
            f"{name} 不是1自由度关节, nv={joint.nv}"
        )

    arm_v_indices.append(
        int(joint.idx_v)
    )

arm_v_indices = np.asarray(
    arm_v_indices,
    dtype=int,
)

print(
    "Arm velocity indices 机械臂速度索引:",
    arm_v_indices.tolist(),
)


# ============================================================
# 4. Initial state
# ============================================================

q0 = pin.neutral(model)
v0 = np.zeros(nv)

q = q0.copy()
v = v0.copy()


# ============================================================
# 5. Desired posture
#
# 当前 nq = 13, nv = 12。
#
# 所以不能简单地假设：
#
#     q[i] += angle
#
# 特别是 continuous joint 可能使用 cos/sin 形式表示 configuration。
#
# 正确方法是在 nv 维切空间中给一个增量 delta，
# 再用：
#
#     q_ref = integrate(model, q0, delta)
#
# ============================================================

delta = np.zeros(nv)

delta[arm_v_indices[1]] = np.deg2rad(+12.0)
delta[arm_v_indices[2]] = np.deg2rad(-18.0)
delta[arm_v_indices[3]] = np.deg2rad(+10.0)

q_ref = pin.integrate(
    model,
    q0,
    delta,
)

v_ref = np.zeros(nv)
a_ref = np.zeros(nv)

print(
    "Desired arm motion 目标机械臂变化(deg): "
    "[0, +12, -18, +10, 0, 0]"
)


# ============================================================
# 6. TSID inverse-dynamics formulation
# ============================================================

formulation = tsid.InverseDynamicsFormulationAccForce(
    "tsid",
    robot,
    False,
)

# Initialize internal Pinocchio data.
formulation.computeProblemData(
    0.0,
    q,
    v,
)


# ============================================================
# 7. Joint posture task
#
# TaskJointPosture internally creates:
#
#     ddq_des =
#         -Kp * e_q
#         -Kd * e_v
#         +ddq_ref
#
# TSID uses Pinocchio difference() for configuration error,
# so nq != nv is handled correctly.
# ============================================================

posture_task = tsid.TaskJointPosture(
    "task-posture",
    robot,
)

kp_value = 25.0

Kp = kp_value * np.ones(na)
Kd = 2.0 * np.sqrt(kp_value) * np.ones(na)

posture_task.setKp(Kp)
posture_task.setKd(Kd)

# Hold all 12 actuated coordinates.
# Arm joints move to q_ref; gripper joints remain at their initial pose.
posture_task.setMask(
    np.ones(na)
)

POSTURE_WEIGHT = 1.0
POSTURE_PRIORITY = 1

formulation.addMotionTask(
    posture_task,
    POSTURE_WEIGHT,
    POSTURE_PRIORITY,
    0.0,
)


# ============================================================
# 8. Reference sample
#
# TaskJointPosture requires:
#
#     reference value       size = nq_actuated = 13
#     reference derivative  size = na          = 12
#     reference acceleration size = na         = 12
#
# Therefore use the two-size TrajectorySample constructor.
# ============================================================

sample = tsid.TrajectorySample(
    nq,
    nv,
)

sample.pos(q_ref)
sample.vel(v_ref)
sample.acc(a_ref)

posture_task.setReference(
    sample
)


# ============================================================
# 9. HQP solver
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
    "HQP dimensions HQP维度: "
    f"nVar={formulation.nVar}, "
    f"nEq={formulation.nEq}, "
    f"nIn={formulation.nIn}"
)


# ============================================================
# 10. Simulation parameters
# ============================================================

dt = 0.001
duration = 5.0

number_of_steps = int(
    duration / dt
)

print_every_steps = int(
    1.0 / dt
)


# ============================================================
# 11. Statistics
# ============================================================

max_dynamic_residual = 0.0
max_tau = 0.0
max_ddq = 0.0

solver_failed = False

check_data = model.createData()


# ============================================================
# 12. TSID control loop
# ============================================================

for step in range(
    number_of_steps + 1
):

    t = step * dt

    # --------------------------------------------------------
    # A. Build current HQP
    # --------------------------------------------------------

    hqp_data = formulation.computeProblemData(
        t,
        q,
        v,
    )

    # --------------------------------------------------------
    # B. Solve HQP
    # --------------------------------------------------------

    solution = solver.solve(
        hqp_data
    )

    if solution.status != 0:
        print(
            f"\nQP solve failed QP求解失败: "
            f"t={t:.3f}s, status={solution.status}"
        )

        solver_failed = True
        break

    # --------------------------------------------------------
    # C. Read TSID outputs
    # --------------------------------------------------------

    tau = np.asarray(
        formulation.getActuatorForces(
            solution
        )
    ).reshape(-1)

    ddq = np.asarray(
        formulation.getAccelerations(
            solution
        )
    ).reshape(-1)

    # --------------------------------------------------------
    # D. Verify inverse dynamics
    #
    # Fixed-base fully-actuated robot, no external contact:
    #
    #     M(q) ddq + h(q,v) = tau
    #
    # RNEA directly computes the left-hand side.
    # --------------------------------------------------------

    tau_rnea = pin.rnea(
        model,
        check_data,
        q,
        v,
        ddq,
    )

    dynamic_residual = float(
        np.linalg.norm(
            tau - tau_rnea
        )
    )

    max_dynamic_residual = max(
        max_dynamic_residual,
        dynamic_residual,
    )

    max_tau = max(
        max_tau,
        float(np.max(np.abs(tau))),
    )

    max_ddq = max(
        max_ddq,
        float(np.max(np.abs(ddq))),
    )

    # --------------------------------------------------------
    # E. Tracking error
    #
    # difference(q, q_ref) belongs to the nv-dimensional
    # tangent space, so we can directly inspect arm DOFs.
    # --------------------------------------------------------

    configuration_error = pin.difference(
        model,
        q,
        q_ref,
    )

    arm_error = configuration_error[
        arm_v_indices
    ]

    arm_error_deg = np.rad2deg(
        arm_error
    )

    arm_error_norm_deg = float(
        np.linalg.norm(
            arm_error_deg
        )
    )

    arm_velocity_norm = float(
        np.linalg.norm(
            v[arm_v_indices]
        )
    )

    # --------------------------------------------------------
    # F. Compact output every 1 second
    # --------------------------------------------------------

    if step % print_every_steps == 0:

        print(
            "\n"
            "---------- TSID State TSID状态 ----------"
        )

        print(
            f"Time 时间: "
            f"{t:.2f} s"
        )

        print(
            f"Arm posture error "
            f"机械臂姿态误差范数: "
            f"{arm_error_norm_deg:.4f} deg"
        )

        print(
            f"Arm speed "
            f"机械臂速度范数: "
            f"{arm_velocity_norm:.5f} rad/s"
        )

        print(
            f"Max torque "
            f"当前最大关节力矩: "
            f"{np.max(np.abs(tau)):.3f} Nm"
        )

        print(
            f"Max acceleration "
            f"当前最大关节加速度: "
            f"{np.max(np.abs(ddq)):.3f} rad/s^2"
        )

        print(
            f"Dynamics residual "
            f"动力学残差: "
            f"{dynamic_residual:.3e}"
        )

    # --------------------------------------------------------
    # G. Internal state integration
    #
    #     v_mid = v + 0.5 dt ddq
    #     q+    = integrate(q, dt * v_mid)
    #     v+    = v + dt ddq
    #
    # This follows the same idea as the official TSID
    # manipulator joint-space example.
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
            dt * v_mid,
        )

        v = (
            v
            +
            dt
            *
            ddq
        )


# ============================================================
# 13. Final result
# ============================================================

final_error = pin.difference(
    model,
    q,
    q_ref,
)

final_arm_error_deg = np.rad2deg(
    final_error[
        arm_v_indices
    ]
)

final_arm_error_norm_deg = float(
    np.linalg.norm(
        final_arm_error_deg
    )
)

final_arm_speed = float(
    np.linalg.norm(
        v[arm_v_indices]
    )
)

print(
    "\n"
    "========== 80A Result "
    "80A结果 =========="
)

print(
    f"Solver status 求解状态: "
    f"{'FAILED 失败' if solver_failed else 'OPTIMAL 正常'}"
)

print(
    f"Final arm posture error "
    f"最终机械臂姿态误差范数: "
    f"{final_arm_error_norm_deg:.6f} deg"
)

print(
    f"Final arm speed "
    f"最终机械臂速度范数: "
    f"{final_arm_speed:.8f} rad/s"
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

print(
    "Final arm error per joint "
    "最终各机械臂关节误差(deg):"
)

print(
    np.array2string(
        final_arm_error_deg,
        precision=6,
        suppress_small=True,
    )
)
