#!/usr/bin/env python3
"""
83_ur3_tsid_contact_force.py

83 - TSID ContactPoint + contact-force optimization
83 - TSID 点接触与接触力优化

实验目标
--------
1. 使用 TSID ContactPoint 在 robotiq_ft_frame_id 建立一个三维点接触；
2. rigid contact 约束该点的 XYZ 位置不动，但不锁定末端姿态；
3. 将接触力加入 HQP 优化变量；
4. 将期望法向力从 0 N 平滑提升到 5 N；
5. 读取 TSID 优化出来的 contact force；
6. 验证：
       - nVar 从 nv=12 增加为 nv+3=15（3个接触力变量）；
       - 接触点位置保持不动；
       - 法向力跟踪 5 N；
       - 关节力矩随接触力变化；
       - 动力学满足
             M(q)ddq + h(q,v) = tau + Jc^T f_c

重要说明
--------
这是“TSID 内部刚性接触模型实验”，不是 MuJoCo/真机 FT300 闭环实验。

TSID 在这里假设：
    接触已经存在，并且接触点是刚性的。

f_c 是优化器中的“模型接触力”，不是 FT300 的真实测量值。
后续真实力控仍然需要传感器反馈、接触估计和模型误差处理。
"""

from pathlib import Path
import warnings

import numpy as np
import pinocchio as pin
import tsid

warnings.filterwarnings(
    "ignore",
    message=".*to-Python converter.*already registered.*",
)

SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "ur3_ft300_robotiq_force_control.urdf"

if not URDF_PATH.exists():
    raise FileNotFoundError(f"找不到 URDF:\n{URDF_PATH}")

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

FRAME_NAME = "robotiq_ft_frame_id"
frame_id = model.getFrameId(FRAME_NAME)

if frame_id >= model.nframes:
    raise RuntimeError(f"找不到 frame: {FRAME_NAME}")

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
        raise RuntimeError(f"找不到机械臂关节: {name}")

    joint = model.joints[joint_id]

    if joint.nv != 1:
        raise RuntimeError(
            f"{name} 的 nv={joint.nv}，本实验期望单自由度关节"
        )

    arm_v_indices.append(int(joint.idx_v))

arm_v_indices = np.asarray(arm_v_indices, dtype=int)


def frame_pose(q_configuration):
    data = model.createData()
    pin.forwardKinematics(model, data, q_configuration)
    pin.updateFramePlacements(model, data)

    placement = data.oMf[frame_id]

    return pin.SE3(
        placement.rotation.copy(),
        placement.translation.copy(),
    )


def frame_linear_jacobian_world_aligned(q_configuration):
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

    J6 = pin.getFrameJacobian(
        model,
        data,
        frame_id,
        pin.LOCAL_WORLD_ALIGNED,
    )

    return J6[:3, :]


def smoothstep01(x):
    x = float(np.clip(x, 0.0, 1.0))
    return 3.0 * x * x - 2.0 * x * x * x


def desired_force(t):
    if t <= 0.5:
        return 0.0

    if t >= 1.5:
        return 5.0

    s = smoothstep01((t - 0.5) / 1.0)
    return 5.0 * s


q_neutral = pin.neutral(model)

initial_arm_deg = np.array([
    0.0,
    -60.0,
    90.0,
    -30.0,
    45.0,
    0.0,
])

initial_delta = np.zeros(nv)
initial_delta[arm_v_indices] = np.deg2rad(initial_arm_deg)

q = pin.integrate(
    model,
    q_neutral,
    initial_delta,
)

v = np.zeros(nv)

q_reference = q.copy()
H_contact_reference = frame_pose(q)

formulation = tsid.InverseDynamicsFormulationAccForce(
    "tsid",
    robot,
    False,
)

formulation.computeProblemData(
    0.0,
    q,
    v,
)

CONTACT_NAME = "tool-contact"

contact_normal = np.array([
    0.0,
    0.0,
    1.0,
])

friction_coefficient = 0.30
minimum_normal_force = 0.0
maximum_normal_force = 20.0

contact = tsid.ContactPoint(
    CONTACT_NAME,
    robot,
    FRAME_NAME,
    contact_normal,
    friction_coefficient,
    minimum_normal_force,
    maximum_normal_force,
)

if int(contact.n_motion) != 3:
    raise RuntimeError(
        f"ContactPoint n_motion={contact.n_motion}, 期望值为3"
    )

if int(contact.n_force) != 3:
    raise RuntimeError(
        f"ContactPoint n_force={contact.n_force}, 期望值为3"
    )

contact.useLocalFrame(False)

kp_contact = 100.0

contact.setKp(
    kp_contact * np.ones(3)
)

contact.setKd(
    2.0 * np.sqrt(kp_contact) * np.ones(3)
)

contact.setReference(
    H_contact_reference
)

contact.setForceReference(
    np.array([
        0.0,
        0.0,
        0.0,
    ])
)

FORCE_REGULARIZATION_WEIGHT = 1.0

added = formulation.addRigidContact(
    contact,
    FORCE_REGULARIZATION_WEIGHT,
)

if not added:
    raise RuntimeError(
        "addRigidContact() 失败"
    )

posture_task = tsid.TaskJointPosture(
    "task-posture",
    robot,
)

kp_posture = 16.0

posture_task.setKp(
    kp_posture * np.ones(na)
)

posture_task.setKd(
    2.0 * np.sqrt(kp_posture) * np.ones(na)
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

    posture_sample.pos(q_reference)
    posture_sample.vel(np.zeros(nv))
    posture_sample.acc(np.zeros(nv))

posture_task.setReference(
    posture_sample
)

formulation.addMotionTask(
    posture_task,
    1.0e-3,
    1,
    0.0,
)

solver = tsid.SolverHQuadProgFast(
    "qp-solver"
)

solver.resize(
    formulation.nVar,
    formulation.nEq,
    formulation.nIn,
)

print(
    "\n"
    "========== 83 TSID Contact Force "
    "83 TSID接触力优化 =========="
)

print(
    f"Robot dimensions 机器人维度: "
    f"nq={nq}, nv={nv}, na={na}"
)

print(
    f"Contact frame 接触坐标系: "
    f"{FRAME_NAME}"
)

print(
    "Contact type 接触类型: "
    "ContactPoint / 3D point contact 三维点接触"
)

print(
    f"Contact dimensions 接触维度: "
    f"n_motion={int(contact.n_motion)}, "
    f"n_force={int(contact.n_force)}"
)

print(
    "Contact normal 接触法向: "
    "[0, 0, 1] world +Z 世界+Z"
)

print(
    f"Friction coefficient 摩擦系数: "
    f"{friction_coefficient:.2f}"
)

print(
    f"Normal-force range 法向力范围: "
    f"[{minimum_normal_force:.1f}, "
    f"{maximum_normal_force:.1f}] N"
)

print(
    "Force command 力指令: "
    "0 N -> 5 N smooth ramp 平滑上升"
)

print(
    f"HQP dimensions HQP维度: "
    f"nVar={formulation.nVar}, "
    f"nEq={formulation.nEq}, "
    f"nIn={formulation.nIn}"
)

print(
    "Expected variable structure "
    "预期优化变量结构: "
    "[ddq(12), contact_force(3)]"
)

dt = 0.001
duration = 3.0
number_of_steps = int(duration / dt)
print_every_steps = int(0.5 / dt)

max_contact_position_error = 0.0
max_force_error_after_ramp = 0.0
max_dynamic_residual_expected_sign = 0.0
max_tau = 0.0

solver_failed = False

last_normal_force = np.nan
last_force_vector = np.zeros(3)

for step in range(number_of_steps + 1):
    t = step * dt

    force_desired = desired_force(t)

    force_reference = np.array([
        0.0,
        0.0,
        force_desired,
    ])

    contact.setForceReference(
        force_reference
    )

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
            f"\nQP solve failed QP求解失败: "
            f"t={t:.3f}s, status={solution.status}"
        )

        solver_failed = True
        break

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

    if hasattr(
        formulation,
        "getContactForce",
    ):
        force_parameter = np.asarray(
            formulation.getContactForce(
                CONTACT_NAME,
                solution,
            )
        ).reshape(-1)
    else:
        force_parameter = np.asarray(
            formulation.getContactForces(
                CONTACT_NAME,
                solution,
            )
        ).reshape(-1)

    if force_parameter.shape != (
        int(contact.n_force),
    ):
        raise RuntimeError(
            "Contact force variable dimension error: "
            f"{force_parameter.shape}"
        )

    force_generator = np.asarray(
        contact.getForceGeneratorMatrix
    )

    force_world = (
        force_generator
        @
        force_parameter
    )

    normal_force = float(
        contact.getNormalForce(
            force_parameter
        )
    )

    H_current = frame_pose(q)

    contact_position_error = float(
        np.linalg.norm(
            H_current.translation
            -
            H_contact_reference.translation
        )
    )

    tau_without_contact = pin.rnea(
        model,
        model.createData(),
        q,
        v,
        ddq,
    )

    J_linear = frame_linear_jacobian_world_aligned(
        q
    )

    tau_expected = (
        tau_without_contact
        -
        J_linear.T @ force_world
    )

    dynamic_residual = float(
        np.linalg.norm(
            tau
            -
            tau_expected
        )
    )

    opposite_sign_residual = float(
        np.linalg.norm(
            tau
            -
            (
                tau_without_contact
                +
                J_linear.T @ force_world
            )
        )
    )

    force_error = abs(
        normal_force
        -
        force_desired
    )

    max_contact_position_error = max(
        max_contact_position_error,
        contact_position_error,
    )

    if t >= 1.5:
        max_force_error_after_ramp = max(
            max_force_error_after_ramp,
            force_error,
        )

    max_dynamic_residual_expected_sign = max(
        max_dynamic_residual_expected_sign,
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

    last_normal_force = normal_force
    last_force_vector = force_world.copy()

    if step % print_every_steps == 0:
        print(
            "\n"
            "---------- 83 State "
            "83状态 ----------"
        )

        print(
            f"Time 时间: "
            f"{t:.2f} s"
        )

        print(
            f"Desired normal force "
            f"目标法向力: "
            f"{force_desired:.3f} N"
        )

        print(
            f"Optimized normal force "
            f"优化法向力: "
            f"{normal_force:.3f} N"
        )

        print(
            f"Force error "
            f"力误差: "
            f"{force_error:.4f} N"
        )

        print(
            "Optimized force XYZ "
            "优化接触力XYZ(N): "
            f"[{force_world[0]:+.3f}, "
            f"{force_world[1]:+.3f}, "
            f"{force_world[2]:+.3f}]"
        )

        print(
            f"Contact position error "
            f"接触点位置误差: "
            f"{contact_position_error * 1000.0:.6f} mm"
        )

        print(
            f"Arm speed "
            f"机械臂速度范数: "
            f"{np.linalg.norm(v[arm_v_indices]):.6f} rad/s"
        )

        print(
            f"Max torque "
            f"当前最大关节力矩: "
            f"{np.max(np.abs(tau)):.3f} Nm"
        )

        print(
            f"Contact dynamics residual "
            f"接触动力学残差: "
            f"{dynamic_residual:.3e}"
        )

        print(
            f"Opposite-sign residual "
            f"反号残差: "
            f"{opposite_sign_residual:.3e}"
        )

    if step < number_of_steps:
        v_mid = (
            v
            +
            0.5 * dt * ddq
        )

        q = pin.integrate(
            model,
            q,
            dt * v_mid,
        )

        v = (
            v
            +
            dt * ddq
        )

print(
    "\n"
    "========== 83 Result "
    "83结果 =========="
)

print(
    f"Solver status "
    f"求解状态: "
    f"{'FAILED 失败' if solver_failed else 'OPTIMAL 正常'}"
)

print(
    f"Final optimized normal force "
    f"最终优化法向力: "
    f"{last_normal_force:.6f} N"
)

print(
    "Final contact force XYZ "
    "最终接触力XYZ(N): "
    f"[{last_force_vector[0]:+.6f}, "
    f"{last_force_vector[1]:+.6f}, "
    f"{last_force_vector[2]:+.6f}]"
)

print(
    f"Max force error after ramp "
    f"力上升完成后最大误差: "
    f"{max_force_error_after_ramp:.6f} N"
)

print(
    f"Max contact position error "
    f"最大接触点位置误差: "
    f"{max_contact_position_error * 1000.0:.6f} mm"
)

print(
    f"Max contact dynamics residual "
    f"最大接触动力学残差: "
    f"{max_dynamic_residual_expected_sign:.3e}"
)

print(
    f"Max torque "
    f"最大关节力矩: "
    f"{max_tau:.3f} Nm"
)

print(
    "\nInterpretation 解释:"
)

print(
    "1) ContactPoint adds 3 force variables "
    "ContactPoint向HQP增加3个接触力变量"
)

print(
    "2) Rigid contact keeps XYZ fixed "
    "刚性点接触约束XYZ位置不动"
)

print(
    "3) Force reference changes optimized contact force "
    "力参考改变优化得到的接触力"
)

print(
    "4) This force is model-side optimized force, not FT300 measurement "
    "该力是模型优化量，不是FT300实测值"
)
