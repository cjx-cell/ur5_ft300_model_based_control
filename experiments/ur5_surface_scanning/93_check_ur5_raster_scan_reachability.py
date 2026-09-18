#!/usr/bin/env python3
"""93 - UR5 full raster-scan reachability and singularity check.

This is the direct UR5 counterpart of the earlier UR3 reachability test.

It keeps ALL task-side quantities unchanged:
- same workpiece
- same 200 x 120 mm scan region
- same 961 scan poses
- same FT300
- same 120 mm inspection probe
- same full 6D desired orientation from step 89

Only the robot model / installation pose has changed to UR5.

Checks:
1. Sequential 6D inverse kinematics for all 961 poses.
2. TCP position and orientation residuals.
3. MuJoCo joint-position limits.
4. Jacobian minimum singular value and condition-number diagnostics.
5. Adjacent joint-configuration continuity.
6. Unexpected collisions in the workcell.

If all poses are valid, the solved joint path is saved for the dynamic
free-space tracking experiment in step 94.
"""

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


HERE = Path(__file__).resolve().parent

URDF_PATH = HERE / "ur5_ft300_inspection.urdf"
WORKCELL_PATH = HERE / "ur5_ft300_inspection_workcell.xml"
PATH_DATA = HERE / "89_raster_scan_path.npz"

OUTPUT = HERE / "93_ur5_scan_ik_solutions.npz"

FRAME_NAME = "inspection_tip"

MAX_IK_ITER = 200

POSITION_TOL = 2.0e-5
ORIENTATION_TOL = np.deg2rad(0.02)

DLS_LAMBDA_GOOD = 1.0e-4
DLS_LAMBDA_NEAR = 2.0e-2
SIGMA_SWITCH = 0.05

MAX_CONFIGURATION_STEP = 0.12

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
    frozenset(
        [
            "inspection_probe_tip_collision",
            "inspection_workpiece",
        ]
    )
}


def mj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(
        model,
        obj_type,
        int(obj_id),
    )
    return (
        name
        if name is not None
        else f"<id:{obj_id}>"
    )


def mujoco_to_pinocchio_q(
    q_mj,
    mj_model,
    pin_model,
):
    q_pin = pin.neutral(
        pin_model
    )

    for pin_joint_id in range(
        1,
        pin_model.njoints,
    ):
        name = pin_model.names[
            pin_joint_id
        ]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if mj_joint_id < 0:
            raise RuntimeError(
                f"MuJoCo缺少关节: {name}"
            )

        angle = q_mj[
            mj_model.jnt_qposadr[
                mj_joint_id
            ]
        ]

        qadr_pin = pin_model.idx_qs[
            pin_joint_id
        ]

        joint = pin_model.joints[
            pin_joint_id
        ]

        if joint.nq == 1:
            q_pin[
                qadr_pin
            ] = angle

        elif (
            joint.nq == 2
            and joint.nv == 1
        ):
            q_pin[
                qadr_pin
            ] = np.cos(
                angle
            )
            q_pin[
                qadr_pin + 1
            ] = np.sin(
                angle
            )

        else:
            raise RuntimeError(
                f"不支持的Pinocchio关节配置表示: "
                f"{name}"
            )

    return q_pin


def pinocchio_to_mujoco_q(
    q_pin,
    pin_model,
    mj_model,
):
    q_mj = np.zeros(
        mj_model.nq
    )

    for pin_joint_id in range(
        1,
        pin_model.njoints,
    ):
        name = pin_model.names[
            pin_joint_id
        ]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if mj_joint_id < 0:
            raise RuntimeError(
                f"MuJoCo缺少关节: {name}"
            )

        qadr_mj = int(
            mj_model.jnt_qposadr[
                mj_joint_id
            ]
        )

        qadr_pin = int(
            pin_model.idx_qs[
                pin_joint_id
            ]
        )

        joint = pin_model.joints[
            pin_joint_id
        ]

        if joint.nq == 1:
            q_mj[
                qadr_mj
            ] = q_pin[
                qadr_pin
            ]

        elif (
            joint.nq == 2
            and joint.nv == 1
        ):
            c = q_pin[
                qadr_pin
            ]
            s = q_pin[
                qadr_pin + 1
            ]

            q_mj[
                qadr_mj
            ] = np.arctan2(
                s,
                c,
            )

        else:
            raise RuntimeError(
                f"不支持的Pinocchio关节配置表示: "
                f"{name}"
            )

    return q_mj


def pose_error(
    pin_model,
    pin_data,
    frame_id,
    q,
    p_des,
    R_des,
):
    pin.forwardKinematics(
        pin_model,
        pin_data,
        q,
    )

    pin.updateFramePlacements(
        pin_model,
        pin_data,
    )

    placement = pin_data.oMf[
        frame_id
    ]

    p = placement.translation
    R = placement.rotation

    e_position = (
        p_des - p
    )

    e_orientation = pin.log3(
        R_des
        @ R.T
    )

    return (
        e_position,
        e_orientation,
    )


def frame_jacobian(
    pin_model,
    pin_data,
    frame_id,
    q,
):
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


def jacobian_diagnostics(
    J,
):
    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )

    sigma_max = float(
        singular_values[0]
    )

    sigma_min = float(
        singular_values[-1]
    )

    condition = (
        sigma_max / sigma_min
        if sigma_min > 1e-12
        else np.inf
    )

    return (
        sigma_min,
        condition,
    )


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

    for iteration in range(
        MAX_IK_ITER
    ):
        (
            e_position,
            e_orientation,
        ) = pose_error(
            pin_model,
            pin_data,
            frame_id,
            q,
            p_des,
            R_des,
        )

        position_error = float(
            np.linalg.norm(
                e_position
            )
        )

        orientation_error = float(
            np.linalg.norm(
                e_orientation
            )
        )

        J = frame_jacobian(
            pin_model,
            pin_data,
            frame_id,
            q,
        )

        (
            sigma_min,
            condition,
        ) = jacobian_diagnostics(
            J
        )

        if (
            position_error
            <= POSITION_TOL
            and orientation_error
            <= ORIENTATION_TOL
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
            if sigma_min
            > SIGMA_SWITCH
            else DLS_LAMBDA_NEAR
        )

        error = np.concatenate(
            [
                e_position,
                e_orientation,
            ]
        )

        matrix = (
            J @ J.T
            + damping
            * damping
            * np.eye(6)
        )

        delta = (
            J.T
            @ np.linalg.solve(
                matrix,
                error,
            )
        )

        delta_norm = float(
            np.linalg.norm(
                delta
            )
        )

        if (
            delta_norm
            > MAX_CONFIGURATION_STEP
        ):
            delta *= (
                MAX_CONFIGURATION_STEP
                / delta_norm
            )

        q = pin.integrate(
            pin_model,
            q,
            delta,
        )

        final_position_error = (
            position_error
        )
        final_orientation_error = (
            orientation_error
        )
        final_sigma_min = (
            sigma_min
        )
        final_condition = (
            condition
        )

    return (
        False,
        q,
        MAX_IK_ITER,
        final_position_error,
        final_orientation_error,
        final_sigma_min,
        final_condition,
    )


def joint_limit_violations(
    q_mj,
    mj_model,
):
    violations = []

    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise RuntimeError(
                f"缺少MuJoCo关节: "
                f"{name}"
            )

        if not bool(
            mj_model.jnt_limited[
                joint_id
            ]
        ):
            continue

        qadr = int(
            mj_model.jnt_qposadr[
                joint_id
            ]
        )

        value = float(
            q_mj[qadr]
        )

        lower = float(
            mj_model.jnt_range[
                joint_id,
                0,
            ]
        )

        upper = float(
            mj_model.jnt_range[
                joint_id,
                1,
            ]
        )

        if (
            value
            < lower - 1e-9
            or value
            > upper + 1e-9
        ):
            violations.append(
                (
                    name,
                    value,
                    lower,
                    upper,
                )
            )

    return violations


def unexpected_contacts(
    mj_model,
    mj_data,
):
    bad_pairs = []

    for contact_id in range(
        mj_data.ncon
    ):
        contact = mj_data.contact[
            contact_id
        ]

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

        pair = frozenset(
            [
                name1,
                name2,
            ]
        )

        if (
            pair
            not in INTENDED_CONTACT_GEOMS
        ):
            bad_pairs.append(
                tuple(
                    sorted(
                        [
                            name1,
                            name2,
                        ]
                    )
                )
            )

    return bad_pairs


def main():
    for path in [
        URDF_PATH,
        WORKCELL_PATH,
        PATH_DATA,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    path_data = np.load(
        PATH_DATA
    )

    positions = path_data[
        "position"
    ]

    rotations = path_data[
        "rotation"
    ]

    arc_length = path_data[
        "arc_length"
    ]

    pin_model = (
        pin.buildModelFromUrdf(
            str(URDF_PATH)
        )
    )

    pin_data = (
        pin_model.createData()
    )

    frame_id = (
        pin_model.getFrameId(
            FRAME_NAME
        )
    )

    if (
        frame_id
        >= pin_model.nframes
    ):
        raise RuntimeError(
            f"Pinocchio找不到frame: "
            f"{FRAME_NAME}"
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

    q_previous = (
        mujoco_to_pinocchio_q(
            mj_data.qpos,
            mj_model,
            pin_model,
        )
    )

    solved_configurations = []
    iteration_counts = []
    position_errors = []
    orientation_errors = []
    sigma_mins = []
    conditions = []
    configuration_steps = []

    failed_indices = []
    limit_violations_all = []

    unexpected_contact_indices = []
    unexpected_contact_examples = []

    print(
        "========== 93 UR5 Reachability "
        "93 UR5全扫描路径可达性 =========="
    )

    print(
        f"Path poses 路径位姿数: "
        f"{len(positions)}"
    )

    print(
        f"Path length 路径长度: "
        f"{arc_length[-1] * 1000.0:.1f} mm"
    )

    print(
        "Comparison condition 对比条件: "
        "same 961 poses as UR3 "
        "与UR3完全相同的961个位姿"
    )

    for index in range(
        len(positions)
    ):
        p_des = positions[
            index
        ]

        R_des = rotations[
            index
        ]

        (
            success,
            q_solution,
            iterations,
            position_error,
            orientation_error,
            sigma_min,
            condition,
        ) = solve_pose_ik(
            pin_model,
            pin_data,
            frame_id,
            q_previous,
            p_des,
            R_des,
        )

        if not success:
            failed_indices.append(
                index
            )

            print(
                f"IK failed IK失败: "
                f"index={index}, "
                f"s={arc_length[index] * 1000.0:.1f} mm, "
                f"pos={position_error * 1000.0:.3f} mm, "
                f"ori={np.degrees(orientation_error):.3f} deg"
            )

            break

        if solved_configurations:
            delta_q = pin.difference(
                pin_model,
                solved_configurations[-1],
                q_solution,
            )

            configuration_steps.append(
                float(
                    np.linalg.norm(
                        delta_q
                    )
                )
            )

        solved_configurations.append(
            q_solution.copy()
        )

        iteration_counts.append(
            iterations
        )

        position_errors.append(
            position_error
        )

        orientation_errors.append(
            orientation_error
        )

        sigma_mins.append(
            sigma_min
        )

        conditions.append(
            condition
        )

        q_mj = pinocchio_to_mujoco_q(
            q_solution,
            pin_model,
            mj_model,
        )

        violations = (
            joint_limit_violations(
                q_mj,
                mj_model,
            )
        )

        if violations:
            limit_violations_all.append(
                (
                    index,
                    violations,
                )
            )

        mj_data.qpos[:] = (
            q_mj
        )

        mj_data.qvel[:] = 0.0

        mujoco.mj_forward(
            mj_model,
            mj_data,
        )

        bad_contacts = (
            unexpected_contacts(
                mj_model,
                mj_data,
            )
        )

        if bad_contacts:
            unexpected_contact_indices.append(
                index
            )

            if (
                len(
                    unexpected_contact_examples
                )
                < 5
            ):
                unexpected_contact_examples.append(
                    (
                        index,
                        bad_contacts,
                    )
                )

        q_previous = q_solution

        if (
            index == 0
            or (
                index + 1
            )
            % PRINT_EVERY
            == 0
            or index
            == len(positions) - 1
        ):
            print(
                f"Progress 进度: "
                f"{index + 1:4d}/{len(positions)} | "
                f"pos={position_error * 1000.0:.3f} mm | "
                f"ori={np.degrees(orientation_error):.3f} deg | "
                f"sigma_min={sigma_min:.4f} | "
                f"cond={condition:.2f}"
            )

    solved_count = len(
        solved_configurations
    )

    if solved_count > 0:
        q_array = np.stack(
            solved_configurations,
            axis=0,
        )

        np.savez(
            OUTPUT,
            q=q_array,
            position=positions[
                :solved_count
            ],
            rotation=rotations[
                :solved_count
            ],
            arc_length=arc_length[
                :solved_count
            ],
            position_error=np.asarray(
                position_errors
            ),
            orientation_error=np.asarray(
                orientation_errors
            ),
            sigma_min=np.asarray(
                sigma_mins
            ),
            condition=np.asarray(
                conditions
            ),
        )

    print(
        "\n========== 93 Result 93结果 =========="
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
            f"最大姿态残差: "
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
            f"Max IK iterations "
            f"单点最大IK迭代次数: "
            f"{max(iteration_counts)}"
        )

    if configuration_steps:
        print(
            f"Max adjacent configuration step "
            f"相邻位姿最大关节配置变化: "
            f"{max(configuration_steps):.5f} rad"
        )

    print(
        f"Joint-limit violations "
        f"关节限位违规: "
        f"{len(limit_violations_all)}"
    )

    print(
        f"Unexpected-collision poses "
        f"非预期碰撞位姿数: "
        f"{len(unexpected_contact_indices)}"
    )

    if unexpected_contact_examples:
        print(
            "Unexpected contact examples "
            "非预期碰撞示例:"
        )

        for (
            index,
            pairs,
        ) in unexpected_contact_examples:
            print(
                f"  index={index}: "
                f"{pairs}"
            )

    all_pass = (
        solved_count
        == len(positions)
        and len(
            limit_violations_all
        )
        == 0
        and len(
            unexpected_contact_indices
        )
        == 0
    )

    print(
        "\nUR3 reference UR3对照结果:"
    )

    print(
        "  IK solved: 114/961"
    )

    print(
        "  min sigma_min: 0.010282"
    )

    print(
        "  max condition: 174.81"
    )

    if all_pass:
        print(
            f"Saved 保存: "
            f"{OUTPUT.name}"
        )

        print(
            "Result 结果: PASS 通过"
        )

        print(
            "Interpretation 解释: "
            "UR5 completed the same full scan task under the same "
            "workpiece/path definition. "
            "UR5在相同工件和扫描轨迹条件下完成完整可达性验证。"
        )

    else:
        print(
            "Result 结果: "
            "CHECK NEEDED 需要调整"
        )

        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()
