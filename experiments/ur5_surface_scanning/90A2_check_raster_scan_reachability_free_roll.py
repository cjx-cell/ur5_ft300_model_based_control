#!/usr/bin/env python3
"""90A2 - Raster-scan reachability with free roll about the probe axis.

Why this version exists
-----------------------
The inspection probe is axisymmetric. For the scanning task we truly need:

    1) TCP position on the surface
    2) tool +Z aligned with the inward surface normal

Rotation ABOUT tool +Z is not task-critical. The original 90A fixed that roll
angle anyway, which can unnecessarily drive the 6-DOF UR3 toward a wrist
singularity.

This script keeps the same TCP position and tool-axis direction, but searches a
small set of roll angles about local tool +Z whenever the current orientation
branch becomes poorly conditioned.

It checks:
- 6D IK convergence for the chosen full pose
- joint limits
- unexpected collisions
- Jacobian sigma_min / condition diagnostics
- adjacent joint-configuration continuity
- chosen roll continuity

If successful, it saves a smooth, fully specified pose path for 90B.
"""

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


HERE = Path(__file__).resolve().parent

URDF_PATH = HERE / "ur3_ft300_inspection.urdf"
WORKCELL_PATH = HERE / "ur3_ft300_inspection_workcell.xml"
PATH_DATA = HERE / "89_raster_scan_path.npz"

OUTPUT = HERE / "90A2_scan_pose_optimized.npz"

FRAME_NAME = "inspection_tip"

MAX_IK_ITER = 160
POSITION_TOL = 2.0e-5
ORIENTATION_TOL = np.deg2rad(0.02)

DLS_LAMBDA_GOOD = 1.0e-4
DLS_LAMBDA_NEAR = 2.0e-2
SIGMA_SWITCH = 0.05

MAX_IK_STEP = 0.10

# If the previous roll still gives a comfortable Jacobian, keep it.
KEEP_SIGMA_MIN = 0.060
KEEP_CONDITION_MAX = 40.0
KEEP_Q_STEP_MAX = 0.12

# Search locally first. Only use the wider angles if necessary.
LOCAL_ROLL_OFFSETS_DEG = [
    0.0,
    +5.0, -5.0,
    +10.0, -10.0,
    +15.0, -15.0,
    +20.0, -20.0,
    +30.0, -30.0,
    +45.0, -45.0,
    +60.0, -60.0,
]

WIDE_ROLL_OFFSETS_DEG = [
    +75.0, -75.0,
    +90.0, -90.0,
    +120.0, -120.0,
    +150.0, -150.0,
    +180.0,
]

PRINT_EVERY = 100

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

INTENDED_CONTACT_GEOMS = {
    frozenset([
        "inspection_probe_tip_collision",
        "inspection_workpiece",
    ])
}


def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def mj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return name if name is not None else f"<id:{obj_id}>"


def pinocchio_to_mujoco_q(q_pin, pin_model, mj_model):
    q_mj = np.zeros(mj_model.nq)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if mj_joint_id < 0:
            raise RuntimeError(f"MuJoCo缺少关节: {name}")

        qadr_mj = int(mj_model.jnt_qposadr[mj_joint_id])
        qadr_pin = int(pin_model.idx_qs[pin_joint_id])
        joint = pin_model.joints[pin_joint_id]

        if joint.nq == 1:
            q_mj[qadr_mj] = q_pin[qadr_pin]

        elif joint.nq == 2 and joint.nv == 1:
            c = q_pin[qadr_pin]
            s = q_pin[qadr_pin + 1]
            q_mj[qadr_mj] = np.arctan2(s, c)

        else:
            raise RuntimeError(
                f"不支持的Pinocchio关节配置表示: {name}"
            )

    return q_mj


def mujoco_to_pinocchio_q(q_mj, mj_model, pin_model):
    q_pin = pin.neutral(pin_model)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if mj_joint_id < 0:
            raise RuntimeError(f"MuJoCo缺少关节: {name}")

        angle = q_mj[mj_model.jnt_qposadr[mj_joint_id]]
        qadr_pin = pin_model.idx_qs[pin_joint_id]
        joint = pin_model.joints[pin_joint_id]

        if joint.nq == 1:
            q_pin[qadr_pin] = angle

        elif joint.nq == 2 and joint.nv == 1:
            q_pin[qadr_pin] = np.cos(angle)
            q_pin[qadr_pin + 1] = np.sin(angle)

        else:
            raise RuntimeError(
                f"不支持的Pinocchio关节配置表示: {name}"
            )

    return q_pin


def pose_error(pin_model, pin_data, frame_id, q, p_des, R_des):
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    placement = pin_data.oMf[frame_id]

    e_position = p_des - placement.translation
    e_orientation = pin.log3(
        R_des @ placement.rotation.T
    )

    return e_position, e_orientation


def frame_jacobian(pin_model, pin_data, frame_id, q):
    pin.computeJointJacobians(
        pin_model,
        pin_data,
        q,
    )
    pin.updateFramePlacements(
        pin_model,
        pin_data,
    )

    return pin.getFrameJacobian(
        pin_model,
        pin_data,
        frame_id,
        pin.LOCAL_WORLD_ALIGNED,
    )


def jacobian_diagnostics(J):
    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )

    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])

    condition = (
        sigma_max / sigma_min
        if sigma_min > 1e-12
        else np.inf
    )

    return sigma_min, condition


def solve_pose_ik(
    pin_model,
    pin_data,
    frame_id,
    q_seed,
    p_des,
    R_des,
):
    q = q_seed.copy()

    final_position_error = np.inf
    final_orientation_error = np.inf
    final_sigma_min = 0.0
    final_condition = np.inf

    for iteration in range(MAX_IK_ITER):
        e_position, e_orientation = pose_error(
            pin_model,
            pin_data,
            frame_id,
            q,
            p_des,
            R_des,
        )

        position_error = float(
            np.linalg.norm(e_position)
        )
        orientation_error = float(
            np.linalg.norm(e_orientation)
        )

        J = frame_jacobian(
            pin_model,
            pin_data,
            frame_id,
            q,
        )

        sigma_min, condition = (
            jacobian_diagnostics(J)
        )

        if (
            position_error <= POSITION_TOL
            and orientation_error <= ORIENTATION_TOL
        ):
            return (
                True,
                q,
                iteration,
                position_error,
                orientation_error,
                sigma_min,
                condition,
            )

        damping = (
            DLS_LAMBDA_GOOD
            if sigma_min > SIGMA_SWITCH
            else DLS_LAMBDA_NEAR
        )

        error = np.concatenate([
            e_position,
            e_orientation,
        ])

        matrix = (
            J @ J.T
            + damping * damping * np.eye(6)
        )

        delta = (
            J.T
            @ np.linalg.solve(
                matrix,
                error,
            )
        )

        delta_norm = float(
            np.linalg.norm(delta)
        )

        if delta_norm > MAX_IK_STEP:
            delta *= (
                MAX_IK_STEP / delta_norm
            )

        q = pin.integrate(
            pin_model,
            q,
            delta,
        )

        final_position_error = position_error
        final_orientation_error = orientation_error
        final_sigma_min = sigma_min
        final_condition = condition

    return (
        False,
        q,
        MAX_IK_ITER,
        final_position_error,
        final_orientation_error,
        final_sigma_min,
        final_condition,
    )


def joint_limit_violations(q_mj, mj_model):
    violations = []

    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise RuntimeError(
                f"缺少MuJoCo关节: {name}"
            )

        if not bool(
            mj_model.jnt_limited[joint_id]
        ):
            continue

        qadr = int(
            mj_model.jnt_qposadr[joint_id]
        )
        value = float(q_mj[qadr])

        lower = float(
            mj_model.jnt_range[joint_id, 0]
        )
        upper = float(
            mj_model.jnt_range[joint_id, 1]
        )

        if (
            value < lower - 1e-9
            or value > upper + 1e-9
        ):
            violations.append(
                (name, value, lower, upper)
            )

    return violations


def unexpected_contacts(mj_model, mj_data):
    bad_pairs = []

    for contact_id in range(mj_data.ncon):
        contact = mj_data.contact[contact_id]

        name1 = mj_name(
            mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )
        name2 = mj_name(
            mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        pair = frozenset([name1, name2])

        if pair not in INTENDED_CONTACT_GEOMS:
            bad_pairs.append(
                tuple(sorted([name1, name2]))
            )

    return bad_pairs


def evaluate_candidate(
    phi,
    base_rotation,
    q_seed,
    p_des,
    pin_model,
    pin_data,
    frame_id,
    mj_model,
    mj_data,
):
    # Right multiplication rotates the tool around its OWN local +Z axis.
    # The third column therefore stays unchanged.
    R_des = (
        base_rotation
        @ rotation_z(phi)
    )

    result = solve_pose_ik(
        pin_model,
        pin_data,
        frame_id,
        q_seed,
        p_des,
        R_des,
    )

    (
        success,
        q_solution,
        iterations,
        position_error,
        orientation_error,
        sigma_min,
        condition,
    ) = result

    if not success:
        return None

    q_mj = pinocchio_to_mujoco_q(
        q_solution,
        pin_model,
        mj_model,
    )

    violations = joint_limit_violations(
        q_mj,
        mj_model,
    )

    if violations:
        return None

    mj_data.qpos[:] = q_mj
    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    bad_contacts = unexpected_contacts(
        mj_model,
        mj_data,
    )

    if bad_contacts:
        return None

    q_step = float(
        np.linalg.norm(
            pin.difference(
                pin_model,
                q_seed,
                q_solution,
            )
        )
    )

    return {
        "phi": phi,
        "R_des": R_des,
        "q": q_solution,
        "iterations": iterations,
        "position_error": position_error,
        "orientation_error": orientation_error,
        "sigma_min": sigma_min,
        "condition": condition,
        "q_step": q_step,
    }


def choose_candidate(
    base_phi,
    offsets_deg,
    base_rotation,
    q_seed,
    p_des,
    pin_model,
    pin_data,
    frame_id,
    mj_model,
    mj_data,
):
    candidates = []

    for offset_deg in offsets_deg:
        phi = (
            base_phi
            + np.deg2rad(offset_deg)
        )

        candidate = evaluate_candidate(
            phi,
            base_rotation,
            q_seed,
            p_des,
            pin_model,
            pin_data,
            frame_id,
            mj_model,
            mj_data,
        )

        if candidate is None:
            continue

        roll_change = abs(
            wrap_to_pi(phi - base_phi)
        )

        candidate["roll_change"] = (
            roll_change
        )

        candidates.append(candidate)

    if not candidates:
        return None

    # Prefer continuity first, but reward a healthier Jacobian.
    # Candidates with very large adjacent joint jumps are penalized strongly.
    def score(candidate):
        return (
            8.0 * candidate["sigma_min"]
            - 0.90 * candidate["q_step"]
            - 0.08 * candidate["roll_change"]
        )

    return max(
        candidates,
        key=score,
    )


def main():
    for path in [
        URDF_PATH,
        WORKCELL_PATH,
        PATH_DATA,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    path_data = np.load(PATH_DATA)

    positions = path_data["position"]
    base_rotations = path_data["rotation"]
    normals = path_data["surface_normal"]
    arc_length = path_data["arc_length"]

    pin_model = pin.buildModelFromUrdf(
        str(URDF_PATH)
    )
    pin_data = pin_model.createData()

    frame_id = pin_model.getFrameId(
        FRAME_NAME
    )

    if frame_id >= pin_model.nframes:
        raise RuntimeError(
            f"Pinocchio找不到frame: {FRAME_NAME}"
        )

    mj_model = mujoco.MjModel.from_xml_path(
        str(WORKCELL_PATH)
    )
    mj_data = mujoco.MjData(
        mj_model
    )

    home_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_id < 0:
        raise RuntimeError(
            "MuJoCo找不到home keyframe"
        )

    mujoco.mj_resetDataKeyframe(
        mj_model,
        mj_data,
        home_id,
    )
    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    q_previous = mujoco_to_pinocchio_q(
        mj_data.qpos,
        mj_model,
        pin_model,
    )

    phi_previous = 0.0

    q_solutions = []
    optimized_rotations = []
    chosen_roll = []
    iteration_counts = []
    position_errors = []
    orientation_errors = []
    sigma_mins = []
    conditions = []
    configuration_steps = []

    roll_adjustment_count = 0
    wide_search_count = 0

    print(
        "========== 90A2 Free-Roll Reachability "
        "90A2自由绕工具轴可达性 =========="
    )
    print(
        f"Path poses 路径位姿数: "
        f"{len(positions)}"
    )
    print(
        "Task definition 任务定义: "
        "TCP position + tool +Z normal alignment; "
        "roll about +Z is free "
        "TCP位置+工具Z轴法向对齐，绕Z轴转角自由"
    )

    for index in range(len(positions)):
        p_des = positions[index]
        base_rotation = base_rotations[index]

        # First test: keep exactly the previous roll.
        candidate_keep = evaluate_candidate(
            phi_previous,
            base_rotation,
            q_previous,
            p_des,
            pin_model,
            pin_data,
            frame_id,
            mj_model,
            mj_data,
        )

        use_keep = False

        if candidate_keep is not None:
            use_keep = (
                candidate_keep["sigma_min"]
                >= KEEP_SIGMA_MIN
                and candidate_keep["condition"]
                <= KEEP_CONDITION_MAX
                and candidate_keep["q_step"]
                <= KEEP_Q_STEP_MAX
            )

        if use_keep:
            candidate = candidate_keep

        else:
            candidate = choose_candidate(
                phi_previous,
                LOCAL_ROLL_OFFSETS_DEG,
                base_rotation,
                q_previous,
                p_des,
                pin_model,
                pin_data,
                frame_id,
                mj_model,
                mj_data,
            )

            if candidate is None:
                wide_search_count += 1

                candidate = choose_candidate(
                    phi_previous,
                    (
                        LOCAL_ROLL_OFFSETS_DEG
                        + WIDE_ROLL_OFFSETS_DEG
                    ),
                    base_rotation,
                    q_previous,
                    p_des,
                    pin_model,
                    pin_data,
                    frame_id,
                    mj_model,
                    mj_data,
                )

            if candidate is not None:
                roll_delta = abs(
                    wrap_to_pi(
                        candidate["phi"]
                        - phi_previous
                    )
                )

                if roll_delta > np.deg2rad(0.1):
                    roll_adjustment_count += 1

        if candidate is None:
            print(
                f"Pose failed 位姿失败: "
                f"index={index}, "
                f"s={arc_length[index] * 1000.0:.1f} mm"
            )
            print(
                "No feasible roll candidate "
                "未找到可行的工具轴滚转角"
            )
            break

        q_solution = candidate["q"]
        R_des = candidate["R_des"]
        phi = candidate["phi"]

        if q_solutions:
            configuration_steps.append(
                float(
                    np.linalg.norm(
                        pin.difference(
                            pin_model,
                            q_solutions[-1],
                            q_solution,
                        )
                    )
                )
            )

        # Verify that free roll did not change tool +Z.
        tool_z = R_des[:, 2]
        inward_normal = -normals[index]

        alignment = float(
            np.clip(
                np.dot(
                    tool_z,
                    inward_normal,
                ),
                -1.0,
                1.0,
            )
        )

        axis_error_deg = np.degrees(
            np.arccos(alignment)
        )

        if axis_error_deg > 1e-4:
            raise RuntimeError(
                f"工具轴法向定义被改变: "
                f"{axis_error_deg:.6e} deg"
            )

        q_solutions.append(
            q_solution.copy()
        )
        optimized_rotations.append(
            R_des.copy()
        )
        chosen_roll.append(phi)
        iteration_counts.append(
            candidate["iterations"]
        )
        position_errors.append(
            candidate["position_error"]
        )
        orientation_errors.append(
            candidate["orientation_error"]
        )
        sigma_mins.append(
            candidate["sigma_min"]
        )
        conditions.append(
            candidate["condition"]
        )

        q_previous = q_solution
        phi_previous = phi

        if (
            index == 0
            or (index + 1) % PRINT_EVERY == 0
            or index == len(positions) - 1
        ):
            print(
                f"Progress 进度: "
                f"{index + 1:4d}/{len(positions)} | "
                f"roll={np.degrees(phi):+.1f} deg | "
                f"sigma_min={candidate['sigma_min']:.4f} | "
                f"cond={candidate['condition']:.2f} | "
                f"dq={candidate['q_step']:.4f} rad"
            )

    solved_count = len(q_solutions)

    if solved_count > 0:
        q_array = np.stack(
            q_solutions,
            axis=0,
        )

        rotation_array = np.stack(
            optimized_rotations,
            axis=0,
        )

        roll_array = np.asarray(
            chosen_roll,
            dtype=float,
        )

        np.savez(
            OUTPUT,
            q=q_array,
            position=positions[:solved_count],
            rotation=rotation_array,
            surface_normal=normals[:solved_count],
            tool_roll=roll_array,
            arc_length=arc_length[:solved_count],
            position_error=np.asarray(position_errors),
            orientation_error=np.asarray(orientation_errors),
            sigma_min=np.asarray(sigma_mins),
            condition=np.asarray(conditions),
        )

    print(
        "\n========== 90A2 Result 90A2结果 =========="
    )

    print(
        f"IK solved IK成功: "
        f"{solved_count}/{len(positions)}"
    )

    if position_errors:
        print(
            f"Max TCP position residual "
            f"最大TCP位置残差: "
            f"{max(position_errors) * 1000.0:.4f} mm"
        )
        print(
            f"Max orientation residual "
            f"最大完整姿态残差: "
            f"{np.degrees(max(orientation_errors)):.4f} deg"
        )
        print(
            f"Minimum sigma_min "
            f"最小雅可比奇异值: "
            f"{min(sigma_mins):.6f}"
        )
        print(
            f"Maximum condition diagnostic "
            f"最大条件数诊断值: "
            f"{max(conditions):.2f}"
        )
        print(
            f"Roll range 工具轴滚转角范围: "
            f"[{np.degrees(min(chosen_roll)):+.1f}, "
            f"{np.degrees(max(chosen_roll)):+.1f}] deg"
        )
        print(
            f"Roll adjustments 滚转角主动调整次数: "
            f"{roll_adjustment_count}"
        )
        print(
            f"Wide searches 宽范围搜索次数: "
            f"{wide_search_count}"
        )

    if configuration_steps:
        print(
            f"Max adjacent configuration step "
            f"相邻位姿最大关节配置变化: "
            f"{max(configuration_steps):.5f} rad"
        )

    all_pass = (
        solved_count == len(positions)
    )

    if all_pass:
        print(
            f"Saved 保存: {OUTPUT.name}"
        )
        print(
            "Result 结果: PASS 通过"
        )
        print(
            "Interpretation 解释: "
            "the probe-axis roll DOF was used only to improve kinematic "
            "conditioning; TCP position and tool-normal alignment were unchanged. "
            "仅利用轴对称探头无关的绕Z轴自由度改善运动学状态，"
            "TCP位置和法向对齐任务保持不变。"
        )
    else:
        print(
            "Result 结果: CHECK NEEDED 需要调整"
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
