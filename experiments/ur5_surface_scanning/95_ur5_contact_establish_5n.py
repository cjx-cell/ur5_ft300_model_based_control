#!/usr/bin/env python3
"""95 - UR5 FT300 contact approach and 5 N force establishment.

This experiment isolates the transition from free-space motion to contact.

Sequence
--------
1. Start 20 mm outside the first raster-scan point.
2. Hold still and calibrate FT300 bias.
3. Move slowly along the local surface normal toward the workpiece.
4. Detect first compressive contact from the FT300 signal.
5. Ramp the normal-force reference to 5 N.
6. Hold 5 N with a 1D admittance correction while maintaining the full tool pose.

Important conventions
---------------------
- n_out: outward workpiece normal.
- tool +Z points inward, so z_tool = -n_out.
- positive normal correction d_n moves the TCP inward:

      p_des = p_surface - d_n * n_out

- FT300 gravity compensation removes the weight of all bodies downstream of
  ft300_site before contact-force estimation.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


HERE = Path(__file__).resolve().parent

MJCF_PATH = HERE / "ur5_ft300_inspection_workcell.xml"
URDF_PATH = HERE / "ur5_ft300_inspection.urdf"
GEOMETRY_PATH = HERE / "89_raster_scan_path.npz"
IK_PATH = HERE / "93_ur5_scan_ik_solutions.npz"

FRAME_NAME = "inspection_tip"
SENSOR_FRAME_NAME = "robotiq_ft_frame_id"

PRECONTACT_GAP = 0.020          # 20 mm outside surface
MAX_APPROACH_PENETRATION = 0.002
APPROACH_DURATION = 6.0

BIAS_START = 0.20
BIAS_END = 0.90

CONTACT_THRESHOLD = 0.25       # N
DESIRED_FORCE = 5.0            # N
FORCE_RAMP_DURATION = 2.0      # s
FORCE_HOLD_DURATION = 5.0      # s

# 1D admittance in the inward normal direction.
FORCE_MASS = 2.0
FORCE_DAMPING = 120.0
NORMAL_VELOCITY_LIMIT = 0.004  # m/s
NORMAL_ACCEL_LIMIT = 0.15      # m/s^2
NORMAL_CORRECTION_MAX = 0.006  # m

# Cartesian pose controller.
KP_LINEAR = 225.0
KD_LINEAR = 30.0
KP_ANGULAR = 144.0
KD_ANGULAR = 24.0

DLS_LAMBDA = 0.03
EXACT_SIGMA_MIN = 0.05
EXACT_CONDITION_MAX = 40.0

FILTER_CUTOFF_HZ = 20.0

PRINT_INTERVAL = 1.0

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

EXPECTED_CONTACT_PAIR = frozenset(
    [
        "inspection_probe_tip_collision",
        "inspection_workpiece",
    ]
)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise RuntimeError("零向量无法归一化")
    return v / n


def quintic(u):
    u = float(np.clip(u, 0.0, 1.0))

    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    sd = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    sdd = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

    return s, sd, sdd


def get_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"找不到模型对象: {name}")
    return idx


def mj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return name if name is not None else f"<id:{obj_id}>"


def sync_pinocchio_numerical_terms(pin_model, mj_model):
    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if mj_joint_id < 0:
            raise RuntimeError(f"MuJoCo缺少关节: {name}")

        pin_v = pin_model.idx_vs[pin_joint_id]
        mj_v = int(mj_model.jnt_dofadr[mj_joint_id])

        pin_model.armature[pin_v] = mj_model.dof_armature[mj_v]
        pin_model.damping[pin_v] = mj_model.dof_damping[mj_v]
        pin_model.friction[pin_v] = mj_model.dof_frictionloss[mj_v]


def mj_q_to_pin(q_mj, mj_model, pin_model):
    q_pin = pin.neutral(pin_model)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        q_pin[pin_model.idx_qs[pin_joint_id]] = q_mj[
            mj_model.jnt_qposadr[mj_joint_id]
        ]

    return q_pin


def mj_v_to_pin(v_mj, mj_model, pin_model):
    v_pin = np.zeros(pin_model.nv)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        v_pin[pin_model.idx_vs[pin_joint_id]] = v_mj[
            mj_model.jnt_dofadr[mj_joint_id]
        ]

    return v_pin


def pin_q_to_mj(q_pin, pin_model, mj_model):
    q_mj = np.zeros(mj_model.nq)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        q_mj[mj_model.jnt_qposadr[mj_joint_id]] = q_pin[
            pin_model.idx_qs[pin_joint_id]
        ]

    return q_mj


def pin_tau_to_mj(tau_pin, pin_model, mj_model):
    tau_mj = np.zeros(mj_model.nv)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]

        mj_joint_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        tau_mj[mj_model.jnt_dofadr[mj_joint_id]] = tau_pin[
            pin_model.idx_vs[pin_joint_id]
        ]

    return tau_mj


def solve_pose_ik(
    pin_model,
    pin_data,
    frame_id,
    q_seed,
    p_des,
    R_des,
):
    q = q_seed.copy()

    for _ in range(200):
        pin.forwardKinematics(
            pin_model,
            pin_data,
            q,
        )
        pin.updateFramePlacements(
            pin_model,
            pin_data,
        )

        pose = pin_data.oMf[frame_id]

        e_position = p_des - pose.translation
        e_orientation = pin.log3(
            R_des @ pose.rotation.T
        )

        if (
            np.linalg.norm(e_position) < 2e-6
            and np.linalg.norm(e_orientation) < np.deg2rad(0.002)
        ):
            return q

        pin.computeJointJacobians(
            pin_model,
            pin_data,
            q,
        )
        pin.updateFramePlacements(
            pin_model,
            pin_data,
        )

        J = pin.getFrameJacobian(
            pin_model,
            pin_data,
            frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )

        singular_values = np.linalg.svd(
            J,
            compute_uv=False,
        )

        sigma_min = singular_values[-1]

        damping = (
            1e-4
            if sigma_min > 0.05
            else 0.02
        )

        error = np.concatenate(
            [
                e_position,
                e_orientation,
            ]
        )

        delta = (
            J.T
            @ np.linalg.solve(
                J @ J.T
                + damping**2 * np.eye(6),
                error,
            )
        )

        delta_norm = np.linalg.norm(delta)

        if delta_norm > 0.10:
            delta *= 0.10 / delta_norm

        q = pin.integrate(
            pin_model,
            q,
            delta,
        )

    raise RuntimeError("预接触位姿IK未收敛")


def is_descendant(model, body_id, root_body_id):
    current = int(body_id)

    while current > 0:
        if current == root_body_id:
            return True

        current = int(
            model.body_parentid[current]
        )

    return False


def unexpected_contacts(model, data):
    bad = []

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]

        name1 = mj_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )

        name2 = mj_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        pair = frozenset(
            [
                name1,
                name2,
            ]
        )

        if pair != EXPECTED_CONTACT_PAIR:
            bad.append(
                tuple(
                    sorted(
                        [
                            name1,
                            name2,
                        ]
                    )
                )
            )

    return bad


def main():
    for path in [
        MJCF_PATH,
        URDF_PATH,
        GEOMETRY_PATH,
        IK_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    geometry = np.load(
        GEOMETRY_PATH
    )
    ik_data = np.load(
        IK_PATH
    )

    p_surface = geometry[
        "position"
    ][0].copy()

    n_out = normalize(
        geometry[
            "surface_normal"
        ][0]
    )

    R_des = geometry[
        "rotation"
    ][0].copy()

    # Explicit convention check.
    tool_z_des = R_des[:, 2]
    tool_axis_error = np.degrees(
        np.arccos(
            np.clip(
                np.dot(
                    tool_z_des,
                    -n_out,
                ),
                -1.0,
                1.0,
            )
        )
    )

    if tool_axis_error > 1e-4:
        raise RuntimeError(
            f"工具Z轴与负法向不一致: {tool_axis_error:.6e} deg"
        )

    mj_model = mujoco.MjModel.from_xml_path(
        str(MJCF_PATH)
    )
    mj_data = mujoco.MjData(
        mj_model
    )

    pin_model = pin.buildModelFromUrdf(
        str(URDF_PATH)
    )
    pin_data = pin_model.createData()

    sync_pinocchio_numerical_terms(
        pin_model,
        mj_model,
    )

    frame_id = pin_model.getFrameId(
        FRAME_NAME
    )
    sensor_frame_id = pin_model.getFrameId(
        SENSOR_FRAME_NAME
    )

    if frame_id >= pin_model.nframes:
        raise RuntimeError(
            f"Pinocchio找不到frame: {FRAME_NAME}"
        )

    if sensor_frame_id >= pin_model.nframes:
        raise RuntimeError(
            f"Pinocchio找不到frame: {SENSOR_FRAME_NAME}"
        )

    ft_site_id = get_id(
        mj_model,
        mujoco.mjtObj.mjOBJ_SITE,
        "ft300_site",
    )

    tip_site_id = get_id(
        mj_model,
        mujoco.mjtObj.mjOBJ_SITE,
        "inspection_tip_site",
    )

    force_sensor_id = get_id(
        mj_model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "ft300_force",
    )

    torque_sensor_id = get_id(
        mj_model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "ft300_torque",
    )

    force_sensor_adr = int(
        mj_model.sensor_adr[
            force_sensor_id
        ]
    )

    torque_sensor_adr = int(
        mj_model.sensor_adr[
            torque_sensor_id
        ]
    )

    sensor_body_id = int(
        mj_model.site_bodyid[
            ft_site_id
        ]
    )

    payload_body_ids = [
        body_id
        for body_id in range(
            mj_model.nbody
        )
        if is_descendant(
            mj_model,
            body_id,
            sensor_body_id,
        )
    ]

    payload_mass = float(
        sum(
            mj_model.body_mass[
                body_id
            ]
            for body_id
            in payload_body_ids
        )
    )

    # --------------------------------------------------------
    # Pre-contact IK
    # --------------------------------------------------------
    q_surface = ik_data[
        "q"
    ][0].copy()

    p_pre = (
        p_surface
        + PRECONTACT_GAP
        * n_out
    )

    q_pre = solve_pose_ik(
        pin_model,
        pin_data,
        frame_id,
        q_surface,
        p_pre,
        R_des,
    )

    mj_data.qpos[:] = pin_q_to_mj(
        q_pre,
        pin_model,
        mj_model,
    )
    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    if mj_data.ncon != 0:
        raise RuntimeError(
            f"预接触起始位姿存在碰撞: ncon={mj_data.ncon}"
        )

    dt = float(
        mj_model.opt.timestep
    )

    # --------------------------------------------------------
    # FT300 helpers
    # --------------------------------------------------------
    def read_ft300_raw():
        force = mj_data.sensordata[
            force_sensor_adr
            :
            force_sensor_adr + 3
        ].copy()

        torque = mj_data.sensordata[
            torque_sensor_adr
            :
            torque_sensor_adr + 3
        ].copy()

        return np.concatenate(
            [
                force,
                torque,
            ]
        )

    def payload_gravity_wrench():
        weighted_com = np.zeros(3)

        for body_id in payload_body_ids:
            weighted_com += (
                mj_model.body_mass[
                    body_id
                ]
                * mj_data.xipos[
                    body_id
                ]
            )

        if payload_mass > 1e-12:
            payload_com = (
                weighted_com
                / payload_mass
            )
        else:
            payload_com = (
                mj_data.site_xpos[
                    ft_site_id
                ].copy()
            )

        sensor_position = (
            mj_data.site_xpos[
                ft_site_id
            ].copy()
        )

        gravity_world = np.array(
            mj_model.opt.gravity
        )

        force_world = (
            -payload_mass
            * gravity_world
        )

        torque_world = np.cross(
            payload_com
            - sensor_position,
            force_world,
        )

        R_sensor_world = (
            mj_data.site_xmat[
                ft_site_id
            ]
            .reshape(
                3,
                3,
            )
            .copy()
        )

        return np.concatenate(
            [
                R_sensor_world.T
                @ force_world,
                R_sensor_world.T
                @ torque_world,
            ]
        )

    def sensor_wrench_to_external_world(
        wrench_sensor,
    ):
        R_sensor_world = (
            mj_data.site_xmat[
                ft_site_id
            ]
            .reshape(
                3,
                3,
            )
            .copy()
        )

        return np.concatenate(
            [
                -R_sensor_world
                @ wrench_sensor[:3],
                -R_sensor_world
                @ wrench_sensor[3:],
            ]
        )

    filter_tau = (
        1.0
        / (
            2.0
            * np.pi
            * FILTER_CUTOFF_HZ
        )
    )

    filter_alpha = (
        dt
        / (
            filter_tau
            + dt
        )
    )

    filtered_sensor_wrench = np.zeros(
        6
    )

    bias_sum = np.zeros(
        6
    )
    bias_count = 0
    bias_wrench = None

    # --------------------------------------------------------
    # Force-control states
    # --------------------------------------------------------
    contact_time = None
    force_sign = None

    normal_correction = 0.0
    normal_velocity = 0.0
    normal_acceleration = 0.0

    force_errors_hold = []
    tcp_errors_hold = []
    normal_errors_hold = []

    max_torque = 0.0
    saturation_steps = 0
    control_steps = 0

    last_print_time = -PRINT_INTERVAL
    summary_printed = False

    # --------------------------------------------------------
    # Cartesian inverse-dynamics controller
    # --------------------------------------------------------
    def cartesian_controller(
        p_des,
        R_target,
        v_linear_des,
        a_linear_des,
        omega_des,
        alpha_des,
        external_world_wrench,
    ):
        q_pin = mj_q_to_pin(
            mj_data.qpos,
            mj_model,
            pin_model,
        )

        v_pin = mj_v_to_pin(
            mj_data.qvel,
            mj_model,
            pin_model,
        )

        pin.forwardKinematics(
            pin_model,
            pin_data,
            q_pin,
            v_pin,
        )

        pin.computeJointJacobians(
            pin_model,
            pin_data,
            q_pin,
        )

        pin.computeJointJacobiansTimeVariation(
            pin_model,
            pin_data,
            q_pin,
            v_pin,
        )

        pin.updateFramePlacements(
            pin_model,
            pin_data,
        )

        pose = pin_data.oMf[
            frame_id
        ]

        p = pose.translation.copy()
        R = pose.rotation.copy()

        J = pin.getFrameJacobian(
            pin_model,
            pin_data,
            frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )

        dJ = pin.getFrameJacobianTimeVariation(
            pin_model,
            pin_data,
            frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )

        spatial_velocity = (
            J @ v_pin
        )

        v_linear = spatial_velocity[
            :3
        ]
        omega = spatial_velocity[
            3:
        ]

        e_position = (
            p_des - p
        )

        e_orientation = pin.log3(
            R_target @ R.T
        )

        a_cmd = (
            a_linear_des
            + KP_LINEAR
            * e_position
            + KD_LINEAR
            * (
                v_linear_des
                - v_linear
            )
        )

        alpha_cmd = (
            alpha_des
            + KP_ANGULAR
            * e_orientation
            + KD_ANGULAR
            * (
                omega_des
                - omega
            )
        )

        desired_spatial_acc = np.concatenate(
            [
                a_cmd,
                alpha_cmd,
            ]
        )

        rhs = (
            desired_spatial_acc
            - dJ
            @ v_pin
        )

        singular_values = np.linalg.svd(
            J,
            compute_uv=False,
        )

        sigma_min = float(
            singular_values[-1]
        )

        condition = float(
            singular_values[0]
            / singular_values[-1]
        )

        use_exact = (
            sigma_min
            > EXACT_SIGMA_MIN
            and condition
            < EXACT_CONDITION_MAX
        )

        if use_exact:
            try:
                qdd = np.linalg.solve(
                    J,
                    rhs,
                )
            except np.linalg.LinAlgError:
                use_exact = False

        if not use_exact:
            qdd = (
                J.T
                @ np.linalg.solve(
                    J @ J.T
                    + DLS_LAMBDA**2
                    * np.eye(6),
                    rhs,
                )
            )

        tau_pin = pin.rnea(
            pin_model,
            pin_data,
            q_pin,
            v_pin,
            qdd,
        )

        # External wrench is measured at the FT300/task-side sensor origin.
        J_sensor = pin.getFrameJacobian(
            pin_model,
            pin_data,
            sensor_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )

        tau_pin -= (
            J_sensor.T
            @ external_world_wrench
        )

        tau_mj = pin_tau_to_mj(
            tau_pin,
            pin_model,
            mj_model,
        )

        tau_act = (
            tau_mj
            - mj_data.qfrc_passive[:6]
        )

        saturated = False

        for i in range(6):
            low, high = (
                mj_model.actuator_ctrlrange[
                    i
                ]
            )

            clipped = np.clip(
                tau_act[i],
                low,
                high,
            )

            if abs(
                clipped
                - tau_act[i]
            ) > 1e-12:
                saturated = True

            tau_act[i] = clipped

        mj_data.ctrl[:6] = (
            tau_act
        )

        return (
            p,
            R,
            e_position,
            e_orientation,
            sigma_min,
            condition,
            tau_act,
            saturated,
        )

    print(
        "========== 95 Contact Establishment "
        "95接触建立与5N力控 =========="
    )
    print(
        f"Pre-contact gap 预接触间隙: "
        f"{PRECONTACT_GAP * 1000.0:.1f} mm"
    )
    print(
        f"Contact threshold 接触检测阈值: "
        f"{CONTACT_THRESHOLD:.2f} N"
    )
    print(
        f"Desired normal force 目标法向力: "
        f"{DESIRED_FORCE:.1f} N"
    )
    print(
        f"Payload mass FT300下游负载质量: "
        f"{payload_mass:.4f} kg"
    )
    print(
        "Normal convention 法向约定: "
        "outward normal n_out; tool +Z = -n_out"
    )

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.02, -0.18, 0.82]
        )
        viewer.cam.distance = 1.85
        viewer.cam.azimuth = 142.0
        viewer.cam.elevation = -24.0

        while viewer.is_running():
            step_start = time.time()
            t = float(
                mj_data.time
            )

            # ------------------------------------------------
            # FT300 preprocessing
            # ------------------------------------------------
            raw_wrench = (
                read_ft300_raw()
            )

            gravity_wrench = (
                payload_gravity_wrench()
            )

            if (
                BIAS_START
                <= t
                < BIAS_END
            ):
                bias_sum += (
                    raw_wrench
                    - gravity_wrench
                )
                bias_count += 1

            if (
                bias_wrench is None
                and t >= BIAS_END
            ):
                if bias_count == 0:
                    raise RuntimeError(
                        "没有采集到FT300 bias数据"
                    )

                bias_wrench = (
                    bias_sum
                    / bias_count
                )

                print(
                    "Bias calibration 零偏标定: completed 完成"
                )

            if bias_wrench is None:
                compensated_sensor_wrench = np.zeros(
                    6
                )
            else:
                compensated_sensor_wrench = (
                    raw_wrench
                    - gravity_wrench
                    - bias_wrench
                )

            filtered_sensor_wrench += (
                filter_alpha
                * (
                    compensated_sensor_wrench
                    - filtered_sensor_wrench
                )
            )

            external_world_wrench = (
                sensor_wrench_to_external_world(
                    filtered_sensor_wrench
                )
            )

            projected_force = float(
                np.dot(
                    external_world_wrench[
                        :3
                    ],
                    n_out,
                )
            )

            # Before first contact the sign is not yet needed.
            if force_sign is None:
                measured_normal_force = abs(
                    projected_force
                )
            else:
                measured_normal_force = max(
                    0.0,
                    force_sign
                    * projected_force,
                )

            # ------------------------------------------------
            # Reference / state machine
            # ------------------------------------------------
            desired_force = 0.0

            if t < BIAS_END:
                phase_text = (
                    "bias/hold 零偏标定"
                )

                p_des = p_pre.copy()
                v_des = np.zeros(3)
                a_des = np.zeros(3)

            elif contact_time is None:
                phase_text = (
                    "approach 接近"
                )

                u = (
                    t - BIAS_END
                ) / APPROACH_DURATION

                s, sd, sdd = quintic(
                    u
                )

                start_offset = (
                    PRECONTACT_GAP
                )

                final_offset = (
                    -MAX_APPROACH_PENETRATION
                )

                delta_offset = (
                    final_offset
                    - start_offset
                )

                offset = (
                    start_offset
                    + s
                    * delta_offset
                )

                offset_dot = (
                    sd
                    * delta_offset
                    / APPROACH_DURATION
                )

                offset_ddot = (
                    sdd
                    * delta_offset
                    / (
                        APPROACH_DURATION**2
                    )
                )

                p_des = (
                    p_surface
                    + offset
                    * n_out
                )

                v_des = (
                    offset_dot
                    * n_out
                )

                a_des = (
                    offset_ddot
                    * n_out
                )

                if (
                    bias_wrench is not None
                    and measured_normal_force
                    >= CONTACT_THRESHOLD
                ):
                    contact_time = t

                    force_sign = (
                        1.0
                        if projected_force >= 0.0
                        else -1.0
                    )

                    p_actual_now = (
                        mj_data.site_xpos[
                            tip_site_id
                        ].copy()
                    )

                    normal_correction = max(
                        0.0,
                        -float(
                            np.dot(
                                p_actual_now
                                - p_surface,
                                n_out,
                            )
                        ),
                    )

                    normal_correction = min(
                        normal_correction,
                        NORMAL_CORRECTION_MAX,
                    )

                    normal_velocity = 0.0

                    print(
                        f"Contact detected 检测到接触: "
                        f"t={contact_time:.3f} s, "
                        f"F={measured_normal_force:.3f} N"
                    )

                    print(
                        f"FT300 normal sign FT300法向符号: "
                        f"{force_sign:+.0f}"
                    )

            else:
                elapsed_contact = (
                    t - contact_time
                )

                if (
                    elapsed_contact
                    < FORCE_RAMP_DURATION
                ):
                    phase_text = (
                        "force ramp 力建立"
                    )

                    s, _, _ = quintic(
                        elapsed_contact
                        / FORCE_RAMP_DURATION
                    )

                    desired_force = (
                        DESIRED_FORCE
                        * s
                    )

                else:
                    phase_text = (
                        "force hold 恒力保持"
                    )

                    desired_force = (
                        DESIRED_FORCE
                    )

                force_error = (
                    desired_force
                    - measured_normal_force
                )

                normal_acceleration = (
                    force_error
                    - FORCE_DAMPING
                    * normal_velocity
                ) / FORCE_MASS

                normal_acceleration = np.clip(
                    normal_acceleration,
                    -NORMAL_ACCEL_LIMIT,
                    +NORMAL_ACCEL_LIMIT,
                )

                normal_velocity += (
                    normal_acceleration
                    * dt
                )

                normal_velocity = np.clip(
                    normal_velocity,
                    -NORMAL_VELOCITY_LIMIT,
                    +NORMAL_VELOCITY_LIMIT,
                )

                normal_correction += (
                    normal_velocity
                    * dt
                )

                if normal_correction <= 0.0:
                    normal_correction = 0.0

                    if normal_velocity < 0.0:
                        normal_velocity = 0.0

                if (
                    normal_correction
                    >= NORMAL_CORRECTION_MAX
                ):
                    normal_correction = (
                        NORMAL_CORRECTION_MAX
                    )

                    if normal_velocity > 0.0:
                        normal_velocity = 0.0

                p_des = (
                    p_surface
                    - normal_correction
                    * n_out
                )

                v_des = (
                    -normal_velocity
                    * n_out
                )

                a_des = (
                    -normal_acceleration
                    * n_out
                )

            # ------------------------------------------------
            # 6D Cartesian controller
            # ------------------------------------------------
            (
                p_actual,
                R_actual,
                e_position,
                e_orientation,
                sigma_min,
                condition,
                tau_act,
                saturated,
            ) = cartesian_controller(
                p_des,
                R_des,
                v_des,
                a_des,
                np.zeros(3),
                np.zeros(3),
                external_world_wrench,
            )

            if saturated:
                saturation_steps += 1

            control_steps += 1

            max_torque = max(
                max_torque,
                float(
                    np.max(
                        np.abs(
                            tau_act
                        )
                    )
                ),
            )

            mujoco.mj_step(
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
                raise RuntimeError(
                    f"出现非预期碰撞: {bad_contacts}"
                )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------
            tcp_error = float(
                np.linalg.norm(
                    e_position
                )
            )

            tool_z_actual = (
                R_actual[:, 2]
            )

            normal_alignment = float(
                np.clip(
                    np.dot(
                        tool_z_actual,
                        -n_out,
                    ),
                    -1.0,
                    1.0,
                )
            )

            normal_error_deg = float(
                np.degrees(
                    np.arccos(
                        normal_alignment
                    )
                )
            )

            if (
                contact_time is not None
                and t
                >= contact_time
                + FORCE_RAMP_DURATION
                + 1.0
            ):
                force_errors_hold.append(
                    abs(
                        DESIRED_FORCE
                        - measured_normal_force
                    )
                )

                tcp_errors_hold.append(
                    tcp_error
                )

                normal_errors_hold.append(
                    normal_error_deg
                )

            if (
                t
                - last_print_time
                >= PRINT_INTERVAL
            ):
                last_print_time = t

                print(
                    f"Time 时间: {t:5.1f} s | "
                    f"{phase_text} | "
                    f"F={desired_force:4.2f}/{measured_normal_force:4.2f} N | "
                    f"dn={normal_correction * 1000.0:+.3f} mm | "
                    f"TCP={tcp_error * 1000.0:.3f} mm | "
                    f"normal={normal_error_deg:.3f} deg | "
                    f"tau={np.max(np.abs(tau_act)):.1f} Nm"
                )

            # ------------------------------------------------
            # Failure / completion checks
            # ------------------------------------------------
            if (
                contact_time is None
                and t
                > BIAS_END
                + APPROACH_DURATION
                + 0.25
            ):
                raise RuntimeError(
                    "接近阶段结束仍未检测到接触"
                )

            if (
                contact_time is not None
                and not summary_printed
                and t
                >= contact_time
                + FORCE_RAMP_DURATION
                + FORCE_HOLD_DURATION
            ):
                summary_printed = True

                saturation_ratio = (
                    100.0
                    * saturation_steps
                    / max(
                        control_steps,
                        1,
                    )
                )

                print(
                    "\n========== 95 Result 95结果 =========="
                )

                print(
                    f"Contact time 接触时间: "
                    f"{contact_time:.3f} s"
                )

                if force_errors_hold:
                    print(
                        f"Force mean/max error "
                        f"恒力平均/最大误差: "
                        f"{np.mean(force_errors_hold):.3f} / "
                        f"{np.max(force_errors_hold):.3f} N"
                    )

                    print(
                        f"TCP mean/max error "
                        f"TCP平均/最大误差: "
                        f"{np.mean(tcp_errors_hold) * 1000.0:.3f} / "
                        f"{np.max(tcp_errors_hold) * 1000.0:.3f} mm"
                    )

                    print(
                        f"Tool-normal mean/max "
                        f"工具法向平均/最大误差: "
                        f"{np.mean(normal_errors_hold):.3f} / "
                        f"{np.max(normal_errors_hold):.3f} deg"
                    )

                print(
                    f"Final normal correction "
                    f"最终法向修正: "
                    f"{normal_correction * 1000.0:+.3f} mm"
                )

                print(
                    f"Max actuator torque "
                    f"最大执行器力矩: "
                    f"{max_torque:.3f} Nm"
                )

                print(
                    f"Torque saturation "
                    f"力矩饱和步占比: "
                    f"{saturation_ratio:.3f} %"
                )

                print(
                    "Unexpected contacts 非预期接触: 0"
                )

                print(
                    "Result 结果: "
                    "if force error is small and saturation is zero, PASS 通过"
                )

                print(
                    "Close viewer to exit 关闭窗口结束"
                )

            viewer.sync()

            remaining = (
                dt
                - (
                    time.time()
                    - step_start
                )
            )

            if remaining > 0.0:
                time.sleep(
                    remaining
                )


if __name__ == "__main__":
    main()
