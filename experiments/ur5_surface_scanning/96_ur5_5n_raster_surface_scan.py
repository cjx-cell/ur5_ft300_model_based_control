#!/usr/bin/env python3
"""96 - UR5 full 3D raster scan with 5 N normal-force control.

Sequence
--------
1. Start 20 mm outside the first scan point.
2. Calibrate FT300 bias while holding still.
3. Approach slowly along the first-point surface normal.
4. Detect contact.
5. Ramp the normal-force reference from 0 N to 5 N.
6. Execute the complete 200 x 120 mm, 1.539 m raster path while maintaining
   5 N normal force.
7. Keep tool +Z aligned with the inward surface normal.

The nominal geometric scan path comes from step 89.
The UR5 feasibility result comes from step 93.
The contact-establishment logic is the validated step-95 controller.
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

# ---------------------------------------------------------------------------
# Contact establishment
# ---------------------------------------------------------------------------

PRECONTACT_GAP = 0.020
MAX_APPROACH_PENETRATION = 0.002
APPROACH_DURATION = 6.0

BIAS_START = 0.20
BIAS_END = 0.90

CONTACT_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Force control
# ---------------------------------------------------------------------------

DESIRED_FORCE = 5.0
FORCE_RAMP_DURATION = 2.0
PRE_SCAN_FORCE_HOLD = 1.0
POST_SCAN_FORCE_HOLD = 1.0

FORCE_MASS = 2.0
FORCE_DAMPING = 120.0

NORMAL_VELOCITY_LIMIT = 0.004
NORMAL_ACCEL_LIMIT = 0.15
NORMAL_CORRECTION_MIN = -0.002
NORMAL_CORRECTION_MAX = +0.006

FILTER_CUTOFF_HZ = 20.0

# ---------------------------------------------------------------------------
# Scan timing
# ---------------------------------------------------------------------------

SCAN_AVG_SPEED = 0.040
CONNECTOR_AVG_SPEED = 0.025
MIN_SEGMENT_DURATION = 0.40

# ---------------------------------------------------------------------------
# Cartesian model-based controller
# ---------------------------------------------------------------------------

KP_LINEAR = 225.0
KD_LINEAR = 30.0

KP_ANGULAR = 144.0
KD_ANGULAR = 24.0

EXACT_SIGMA_MIN = 0.05
EXACT_CONDITION_MAX = 40.0
DLS_LAMBDA = 0.03

PRINT_INTERVAL = 1.0

EXPECTED_CONTACT_PAIR = frozenset(
    [
        "inspection_probe_tip_collision",
        "inspection_workpiece",
    ]
)


def normalize(v):
    norm = np.linalg.norm(v)

    if norm < 1e-12:
        raise RuntimeError("零向量无法归一化")

    return v / norm


def quintic(u):
    u = float(
        np.clip(
            u,
            0.0,
            1.0,
        )
    )

    s = (
        10.0 * u**3
        - 15.0 * u**4
        + 6.0 * u**5
    )

    ds = (
        30.0 * u**2
        - 60.0 * u**3
        + 30.0 * u**4
    )

    d2s = (
        60.0 * u
        - 180.0 * u**2
        + 120.0 * u**3
    )

    return s, ds, d2s


def get_id(
    model,
    obj_type,
    name,
):
    idx = mujoco.mj_name2id(
        model,
        obj_type,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"找不到模型对象: {name}"
        )

    return idx


def mj_name(
    model,
    obj_type,
    obj_id,
):
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


def sync_pinocchio_numerical_terms(
    pin_model,
    mj_model,
):
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

        pin_v = pin_model.idx_vs[
            pin_joint_id
        ]

        mj_v = int(
            mj_model.jnt_dofadr[
                mj_joint_id
            ]
        )

        pin_model.armature[
            pin_v
        ] = mj_model.dof_armature[
            mj_v
        ]

        pin_model.damping[
            pin_v
        ] = mj_model.dof_damping[
            mj_v
        ]

        pin_model.friction[
            pin_v
        ] = mj_model.dof_frictionloss[
            mj_v
        ]


def mj_q_to_pin(
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

        q_pin[
            pin_model.idx_qs[
                pin_joint_id
            ]
        ] = q_mj[
            mj_model.jnt_qposadr[
                mj_joint_id
            ]
        ]

    return q_pin


def mj_v_to_pin(
    v_mj,
    mj_model,
    pin_model,
):
    v_pin = np.zeros(
        pin_model.nv
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

        v_pin[
            pin_model.idx_vs[
                pin_joint_id
            ]
        ] = v_mj[
            mj_model.jnt_dofadr[
                mj_joint_id
            ]
        ]

    return v_pin


def pin_q_to_mj(
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

        q_mj[
            mj_model.jnt_qposadr[
                mj_joint_id
            ]
        ] = q_pin[
            pin_model.idx_qs[
                pin_joint_id
            ]
        ]

    return q_mj


def pin_tau_to_mj(
    tau_pin,
    pin_model,
    mj_model,
):
    tau_mj = np.zeros(
        mj_model.nv
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

        tau_mj[
            mj_model.jnt_dofadr[
                mj_joint_id
            ]
        ] = tau_pin[
            pin_model.idx_vs[
                pin_joint_id
            ]
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

    for _ in range(
        200
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

        pose = pin_data.oMf[
            frame_id
        ]

        e_position = (
            p_des
            - pose.translation
        )

        e_orientation = pin.log3(
            R_des
            @ pose.rotation.T
        )

        if (
            np.linalg.norm(
                e_position
            )
            < 2e-6
            and np.linalg.norm(
                e_orientation
            )
            < np.deg2rad(
                0.002
            )
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

        damping = (
            1e-4
            if singular_values[-1]
            > 0.05
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
                + damping**2
                * np.eye(6),
                error,
            )
        )

        delta_norm = np.linalg.norm(
            delta
        )

        if delta_norm > 0.10:
            delta *= (
                0.10
                / delta_norm
            )

        q = pin.integrate(
            pin_model,
            q,
            delta,
        )

    raise RuntimeError(
        "预接触位姿IK未收敛"
    )


def is_descendant(
    model,
    body_id,
    root_body_id,
):
    current = int(
        body_id
    )

    while current > 0:
        if current == root_body_id:
            return True

        current = int(
            model.body_parentid[
                current
            ]
        )

    return False


def unexpected_contacts(
    model,
    data,
):
    bad = []

    for contact_id in range(
        data.ncon
    ):
        contact = data.contact[
            contact_id
        ]

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

        if (
            pair
            != EXPECTED_CONTACT_PAIR
        ):
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


def contiguous_segments(
    row_id,
    phase,
):
    keys = list(
        zip(
            row_id.tolist(),
            phase.tolist(),
        )
    )

    runs = []
    run_start = 0

    for index in range(
        1,
        len(keys),
    ):
        if (
            keys[index]
            != keys[index - 1]
        ):
            runs.append(
                (
                    run_start,
                    index - 1,
                    keys[index - 1],
                )
            )

            run_start = index

    runs.append(
        (
            run_start,
            len(keys) - 1,
            keys[-1],
        )
    )

    segments = []

    for run_index, (
        start,
        end,
        key,
    ) in enumerate(
        runs
    ):
        if run_index > 0:
            start -= 1

        segments.append(
            {
                "start": start,
                "end": end,
                "row": int(
                    key[0]
                ),
                "phase": int(
                    key[1]
                ),
            }
        )

    return segments


def interp_vector(
    x,
    xp,
    values,
):
    result = np.empty(
        values.shape[1]
    )

    for axis in range(
        values.shape[1]
    ):
        result[
            axis
        ] = np.interp(
            x,
            xp,
            values[:, axis],
        )

    return result


def rotation_path_derivative(
    rotations,
    s_local,
):
    count = len(
        rotations
    )

    omega_s = np.zeros(
        (
            count,
            3,
        )
    )

    if count < 2:
        return omega_s

    for index in range(
        count
    ):
        if index == 0:
            ds = (
                s_local[1]
                - s_local[0]
            )

            delta_R = (
                rotations[1]
                @ rotations[0].T
            )

        elif index == (
            count - 1
        ):
            ds = (
                s_local[-1]
                - s_local[-2]
            )

            delta_R = (
                rotations[-1]
                @ rotations[-2].T
            )

        else:
            ds = (
                s_local[index + 1]
                - s_local[index - 1]
            )

            delta_R = (
                rotations[index + 1]
                @ rotations[index - 1].T
            )

        omega_s[
            index
        ] = (
            pin.log3(
                delta_R
            )
            / ds
        )

    return omega_s


def make_segment(
    raw_segment,
    arc_length,
    positions,
    normals,
    rotations,
):
    start = raw_segment[
        "start"
    ]

    end = raw_segment[
        "end"
    ]

    s = arc_length[
        start
        :
        end + 1
    ].copy()

    s_local = (
        s
        - s[0]
    )

    p = positions[
        start
        :
        end + 1
    ].copy()

    n = normals[
        start
        :
        end + 1
    ].copy()

    R = rotations[
        start
        :
        end + 1
    ].copy()

    length = float(
        s_local[-1]
    )

    if length <= 0.0:
        raise RuntimeError(
            "轨迹段长度非正"
        )

    edge_order = (
        2
        if len(s_local) >= 3
        else 1
    )

    p_s = np.gradient(
        p,
        s_local,
        axis=0,
        edge_order=edge_order,
    )

    p_ss = np.gradient(
        p_s,
        s_local,
        axis=0,
        edge_order=edge_order,
    )

    n_s = np.gradient(
        n,
        s_local,
        axis=0,
        edge_order=edge_order,
    )

    n_ss = np.gradient(
        n_s,
        s_local,
        axis=0,
        edge_order=edge_order,
    )

    omega_s = (
        rotation_path_derivative(
            R,
            s_local,
        )
    )

    omega_ss = np.gradient(
        omega_s,
        s_local,
        axis=0,
        edge_order=edge_order,
    )

    speed = (
        SCAN_AVG_SPEED
        if raw_segment[
            "phase"
        ] == 0
        else CONNECTOR_AVG_SPEED
    )

    duration = max(
        length
        / speed,
        MIN_SEGMENT_DURATION,
    )

    return {
        **raw_segment,
        "s": s,
        "s_local": s_local,
        "p": p,
        "n": n,
        "R": R,
        "p_s": p_s,
        "p_ss": p_ss,
        "n_s": n_s,
        "n_ss": n_ss,
        "omega_s": omega_s,
        "omega_ss": omega_ss,
        "length": length,
        "duration": duration,
    }


def interp_rotation(
    x,
    xp,
    rotations,
):
    if x <= xp[0]:
        return rotations[0].copy()

    if x >= xp[-1]:
        return rotations[-1].copy()

    index = int(
        np.searchsorted(
            xp,
            x,
        )
        - 1
    )

    index = max(
        0,
        min(
            index,
            len(xp) - 2,
        ),
    )

    denominator = (
        xp[index + 1]
        - xp[index]
    )

    alpha = (
        x - xp[index]
    ) / denominator

    R0 = rotations[
        index
    ]

    R1 = rotations[
        index + 1
    ]

    delta = pin.log3(
        R1 @ R0.T
    )

    return (
        pin.exp3(
            alpha
            * delta
        )
        @ R0
    )


def segment_reference(
    segment,
    local_time,
):
    duration = segment[
        "duration"
    ]

    u = (
        local_time
        / duration
    )

    sigma, dsigma, d2sigma = quintic(
        u
    )

    length = segment[
        "length"
    ]

    s_local = (
        length
        * sigma
    )

    sdot = (
        length
        * dsigma
        / duration
    )

    sddot = (
        length
        * d2sigma
        / (
            duration**2
        )
    )

    p = interp_vector(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "p"
        ],
    )

    n = normalize(
        interp_vector(
            s_local,
            segment[
                "s_local"
            ],
            segment[
                "n"
            ],
        )
    )

    R = interp_rotation(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "R"
        ],
    )

    p_s = interp_vector(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "p_s"
        ],
    )

    p_ss = interp_vector(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "p_ss"
        ],
    )

    n_s = interp_vector(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "n_s"
        ],
    )

    n_ss = interp_vector(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "n_ss"
        ],
    )

    omega_s = interp_vector(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "omega_s"
        ],
    )

    omega_ss = interp_vector(
        s_local,
        segment[
            "s_local"
        ],
        segment[
            "omega_ss"
        ],
    )

    s_global = (
        segment[
            "s"
        ][0]
        + s_local
    )

    return (
        p,
        n,
        R,
        p_s,
        p_ss,
        n_s,
        n_ss,
        omega_s,
        omega_ss,
        sdot,
        sddot,
        s_global,
    )


def main():
    for path in [
        MJCF_PATH,
        URDF_PATH,
        GEOMETRY_PATH,
        IK_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    geometry = np.load(
        GEOMETRY_PATH
    )

    ik_data = np.load(
        IK_PATH
    )

    positions = geometry[
        "position"
    ].copy()

    normals = geometry[
        "surface_normal"
    ].copy()

    rotations = geometry[
        "rotation"
    ].copy()

    arc_length = geometry[
        "arc_length"
    ].copy()

    row_id = geometry[
        "row_id"
    ].copy()

    phase = geometry[
        "phase"
    ].copy()

    p_surface_start = positions[
        0
    ].copy()

    n_start = normalize(
        normals[
            0
        ]
    )

    R_start = rotations[
        0
    ].copy()

    tool_axis_error = np.degrees(
        np.arccos(
            np.clip(
                np.dot(
                    R_start[:, 2],
                    -n_start,
                ),
                -1.0,
                1.0,
            )
        )
    )

    if tool_axis_error > 1e-4:
        raise RuntimeError(
            "初始工具Z轴与负法向不一致"
        )

    raw_segments = contiguous_segments(
        row_id,
        phase,
    )

    segments = [
        make_segment(
            raw_segment,
            arc_length,
            positions,
            normals,
            rotations,
        )
        for raw_segment
        in raw_segments
    ]

    scan_duration = sum(
        segment[
            "duration"
        ]
        for segment
        in segments
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

    if (
        frame_id
        >= pin_model.nframes
        or sensor_frame_id
        >= pin_model.nframes
    ):
        raise RuntimeError(
            "Pinocchio缺少TCP或FT300 frame"
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

    q_surface_start = ik_data[
        "q"
    ][0].copy()

    p_pre = (
        p_surface_start
        + PRECONTACT_GAP
        * n_start
    )

    q_pre = solve_pose_ik(
        pin_model,
        pin_data,
        frame_id,
        q_surface_start,
        p_pre,
        R_start,
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
        weighted_com = np.zeros(
            3
        )

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

    contact_time = None
    force_sign = None
    scan_start_time = None
    scan_finish_time = None

    normal_correction = 0.0
    normal_velocity = 0.0
    normal_acceleration = 0.0

    segment_index = 0
    segment_elapsed = 0.0

    force_errors_scan = []
    tcp_errors_scan = []
    normal_errors_scan = []
    tangential_forces_scan = []
    normal_corrections_scan = []
    sigma_mins_scan = []
    conditions_scan = []

    max_torque = 0.0
    saturation_steps = 0
    control_steps = 0

    summary_printed = False
    last_print_time = -PRINT_INTERVAL

    def cartesian_controller(
        p_des,
        R_des,
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
            R_des @ R.T
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
            - mj_data.qfrc_passive[
                :6
            ]
        )

        saturated = False

        for actuator_id in range(
            6
        ):
            low, high = (
                mj_model.actuator_ctrlrange[
                    actuator_id
                ]
            )

            clipped = np.clip(
                tau_act[
                    actuator_id
                ],
                low,
                high,
            )

            if abs(
                clipped
                - tau_act[
                    actuator_id
                ]
            ) > 1e-12:
                saturated = True

            tau_act[
                actuator_id
            ] = clipped

        mj_data.ctrl[
            :6
        ] = tau_act

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
        "========== 96 UR5 5N Surface Scan "
        "96 UR5三维曲面5N恒力扫描 =========="
    )

    print(
        f"Path length 路径长度: "
        f"{arc_length[-1] * 1000.0:.1f} mm"
    )

    print(
        f"Scan area 扫描区域: "
        f"200 x 120 mm"
    )

    print(
        f"Segments 轨迹段数: "
        f"{len(segments)}"
    )

    print(
        f"Planned scan time 计划扫描时间: "
        f"{scan_duration:.1f} s"
    )

    print(
        f"Desired force 目标法向力: "
        f"{DESIRED_FORCE:.1f} N"
    )

    print(
        f"Payload mass FT300下游负载质量: "
        f"{payload_mass:.4f} kg"
    )

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [
                0.02,
                -0.18,
                0.82,
            ]
        )

        viewer.cam.distance = 1.85
        viewer.cam.azimuth = 142.0
        viewer.cam.elevation = -24.0

        while viewer.is_running():
            step_wall_start = time.time()

            t = float(
                mj_data.time
            )

            # ------------------------------------------------------------
            # FT300 preprocessing
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # State machine / nominal geometry
            # ------------------------------------------------------------
            desired_force = 0.0

            nominal_position = (
                p_surface_start.copy()
            )

            current_normal = (
                n_start.copy()
            )

            desired_rotation = (
                R_start.copy()
            )

            nominal_velocity = np.zeros(
                3
            )

            nominal_acceleration = np.zeros(
                3
            )

            desired_omega = np.zeros(
                3
            )

            desired_alpha = np.zeros(
                3
            )

            progress_percent = 0.0

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

                s, ds, d2s = quintic(
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
                    ds
                    * delta_offset
                    / APPROACH_DURATION
                )

                offset_ddot = (
                    d2s
                    * delta_offset
                    / (
                        APPROACH_DURATION**2
                    )
                )

                p_des = (
                    p_surface_start
                    + offset
                    * n_start
                )

                v_des = (
                    offset_dot
                    * n_start
                )

                a_des = (
                    offset_ddot
                    * n_start
                )

            else:
                elapsed_contact = (
                    t
                    - contact_time
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

                elif (
                    elapsed_contact
                    < FORCE_RAMP_DURATION
                    + PRE_SCAN_FORCE_HOLD
                ):
                    phase_text = (
                        "force hold 恒力预保持"
                    )

                    desired_force = (
                        DESIRED_FORCE
                    )

                else:
                    if scan_start_time is None:
                        scan_start_time = t

                        segment_index = 0
                        segment_elapsed = 0.0

                        print(
                            f"Scan started 扫描开始: "
                            f"t={scan_start_time:.3f} s"
                        )

                    scan_elapsed = (
                        t
                        - scan_start_time
                    )

                    while (
                        segment_index
                        < len(segments)
                        and scan_elapsed
                        >= segment_elapsed
                        + segments[
                            segment_index
                        ][
                            "duration"
                        ]
                    ):
                        segment_elapsed += (
                            segments[
                                segment_index
                            ][
                                "duration"
                            ]
                        )

                        segment_index += 1

                    if (
                        segment_index
                        < len(segments)
                    ):
                        segment = segments[
                            segment_index
                        ]

                        local_time = (
                            scan_elapsed
                            - segment_elapsed
                        )

                        (
                            nominal_position,
                            current_normal,
                            desired_rotation,
                            p_s,
                            p_ss,
                            n_s,
                            n_ss,
                            omega_s,
                            omega_ss,
                            sdot,
                            sddot,
                            s_global,
                        ) = segment_reference(
                            segment,
                            local_time,
                        )

                        nominal_velocity = (
                            p_s
                            * sdot
                        )

                        nominal_acceleration = (
                            p_ss
                            * sdot**2
                            + p_s
                            * sddot
                        )

                        desired_omega = (
                            omega_s
                            * sdot
                        )

                        desired_alpha = (
                            omega_ss
                            * sdot**2
                            + omega_s
                            * sddot
                        )

                        progress_percent = (
                            100.0
                            * s_global
                            / arc_length[-1]
                        )

                        phase_text = (
                            "scan 扫描"
                            if segment[
                                "phase"
                            ] == 0
                            else "turn 换行"
                        )

                    else:
                        if scan_finish_time is None:
                            scan_finish_time = t

                            print(
                                f"Scan finished 扫描完成: "
                                f"t={scan_finish_time:.3f} s"
                            )

                        phase_text = (
                            "post hold 结束保持"
                        )

                        desired_force = (
                            DESIRED_FORCE
                        )

                        nominal_position = (
                            positions[-1].copy()
                        )

                        current_normal = normalize(
                            normals[-1]
                        )

                        desired_rotation = (
                            rotations[-1].copy()
                        )

                        progress_percent = (
                            100.0
                        )

                    desired_force = (
                        DESIRED_FORCE
                    )

            # ------------------------------------------------------------
            # Normal-force measurement
            # ------------------------------------------------------------
            projected_force = float(
                np.dot(
                    external_world_wrench[
                        :3
                    ],
                    current_normal,
                )
            )

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

            # Contact detection occurs during approach.
            if (
                contact_time is None
                and bias_wrench is not None
                and t >= BIAS_END
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

                normal_correction = (
                    -float(
                        np.dot(
                            p_actual_now
                            - p_surface_start,
                            n_start,
                        )
                    )
                )

                normal_correction = float(
                    np.clip(
                        normal_correction,
                        NORMAL_CORRECTION_MIN,
                        NORMAL_CORRECTION_MAX,
                    )
                )

                normal_velocity = 0.0
                normal_acceleration = 0.0

                print(
                    f"Contact detected 检测到接触: "
                    f"t={contact_time:.3f} s, "
                    f"F={measured_normal_force:.3f} N"
                )

                print(
                    f"FT300 normal sign FT300法向符号: "
                    f"{force_sign:+.0f}"
                )

            # ------------------------------------------------------------
            # Force admittance after contact
            # ------------------------------------------------------------
            if (
                contact_time is not None
            ):
                force_error = (
                    desired_force
                    - measured_normal_force
                )

                normal_acceleration = (
                    force_error
                    - FORCE_DAMPING
                    * normal_velocity
                ) / FORCE_MASS

                normal_acceleration = float(
                    np.clip(
                        normal_acceleration,
                        -NORMAL_ACCEL_LIMIT,
                        +NORMAL_ACCEL_LIMIT,
                    )
                )

                normal_velocity += (
                    normal_acceleration
                    * dt
                )

                normal_velocity = float(
                    np.clip(
                        normal_velocity,
                        -NORMAL_VELOCITY_LIMIT,
                        +NORMAL_VELOCITY_LIMIT,
                    )
                )

                normal_correction += (
                    normal_velocity
                    * dt
                )

                normal_correction = float(
                    np.clip(
                        normal_correction,
                        NORMAL_CORRECTION_MIN,
                        NORMAL_CORRECTION_MAX,
                    )
                )

                if (
                    normal_correction
                    <= NORMAL_CORRECTION_MIN
                    and normal_velocity < 0.0
                ):
                    normal_velocity = 0.0

                if (
                    normal_correction
                    >= NORMAL_CORRECTION_MAX
                    and normal_velocity > 0.0
                ):
                    normal_velocity = 0.0

                # Positive correction moves inward.
                p_des = (
                    nominal_position
                    - normal_correction
                    * current_normal
                )

                # Derivative of:
                # p_des = p_nom - d_n n
                if (
                    phase_text
                    in {
                        "scan 扫描",
                        "turn 换行",
                    }
                ):
                    # Values available from segment_reference.
                    v_des = (
                        nominal_velocity
                        - normal_velocity
                        * current_normal
                        - normal_correction
                        * n_s
                        * sdot
                    )

                    a_des = (
                        nominal_acceleration
                        - normal_acceleration
                        * current_normal
                        - 2.0
                        * normal_velocity
                        * n_s
                        * sdot
                        - normal_correction
                        * (
                            n_ss
                            * sdot**2
                            + n_s
                            * sddot
                        )
                    )

                else:
                    v_des = (
                        -normal_velocity
                        * current_normal
                    )

                    a_des = (
                        -normal_acceleration
                        * current_normal
                    )

            # ------------------------------------------------------------
            # Cartesian inverse dynamics
            # ------------------------------------------------------------
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
                desired_rotation,
                v_des,
                a_des,
                desired_omega,
                desired_alpha,
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

            bad_contacts = unexpected_contacts(
                mj_model,
                mj_data,
            )

            if bad_contacts:
                raise RuntimeError(
                    f"出现非预期碰撞: {bad_contacts}"
                )

            # ------------------------------------------------------------
            # Metrics
            # ------------------------------------------------------------
            tcp_error = float(
                np.linalg.norm(
                    e_position
                )
            )

            tool_z_actual = (
                R_actual[
                    :,
                    2
                ]
            )

            normal_alignment = float(
                np.clip(
                    np.dot(
                        tool_z_actual,
                        -current_normal,
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

            force_vector = (
                external_world_wrench[
                    :3
                ]
            )

            tangential_vector = (
                force_vector
                - np.dot(
                    force_vector,
                    current_normal,
                )
                * current_normal
            )

            tangential_force = float(
                np.linalg.norm(
                    tangential_vector
                )
            )

            if (
                scan_start_time is not None
                and scan_finish_time is None
            ):
                force_errors_scan.append(
                    abs(
                        DESIRED_FORCE
                        - measured_normal_force
                    )
                )

                tcp_errors_scan.append(
                    tcp_error
                )

                normal_errors_scan.append(
                    normal_error_deg
                )

                tangential_forces_scan.append(
                    tangential_force
                )

                normal_corrections_scan.append(
                    normal_correction
                )

                sigma_mins_scan.append(
                    sigma_min
                )

                conditions_scan.append(
                    condition
                )

            # ------------------------------------------------------------
            # Compact print
            # ------------------------------------------------------------
            if (
                t
                - last_print_time
                >= PRINT_INTERVAL
            ):
                last_print_time = t

                print(
                    f"Time 时间: {t:5.1f} s | "
                    f"{phase_text} | "
                    f"progress={progress_percent:5.1f}% | "
                    f"F={desired_force:4.2f}/{measured_normal_force:4.2f} N | "
                    f"dn={normal_correction * 1000.0:+.3f} mm | "
                    f"TCP={tcp_error * 1000.0:.3f} mm | "
                    f"normal={normal_error_deg:.3f} deg | "
                    f"tau={np.max(np.abs(tau_act)):.1f} Nm"
                )

            # ------------------------------------------------------------
            # Safety / completion
            # ------------------------------------------------------------
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
                scan_finish_time is not None
                and not summary_printed
                and t
                >= scan_finish_time
                + POST_SCAN_FORCE_HOLD
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
                    "\n========== 96 Result 96结果 =========="
                )

                print(
                    f"Scan duration 实际扫描时间: "
                    f"{scan_finish_time - scan_start_time:.3f} s"
                )

                if force_errors_scan:
                    print(
                        f"Force mean/max error "
                        f"扫描力平均/最大误差: "
                        f"{np.mean(force_errors_scan):.3f} / "
                        f"{np.max(force_errors_scan):.3f} N"
                    )

                    print(
                        f"TCP mean/max error "
                        f"TCP平均/最大跟踪误差: "
                        f"{np.mean(tcp_errors_scan) * 1000.0:.3f} / "
                        f"{np.max(tcp_errors_scan) * 1000.0:.3f} mm"
                    )

                    print(
                        f"Tool-normal mean/max "
                        f"工具法向平均/最大误差: "
                        f"{np.mean(normal_errors_scan):.3f} / "
                        f"{np.max(normal_errors_scan):.3f} deg"
                    )

                    print(
                        f"Tangential force mean/max "
                        f"切向力平均/最大值: "
                        f"{np.mean(tangential_forces_scan):.3f} / "
                        f"{np.max(tangential_forces_scan):.3f} N"
                    )

                    print(
                        f"Normal correction mean/min/max "
                        f"法向修正平均/最小/最大: "
                        f"{np.mean(normal_corrections_scan) * 1000.0:+.3f} / "
                        f"{np.min(normal_corrections_scan) * 1000.0:+.3f} / "
                        f"{np.max(normal_corrections_scan) * 1000.0:+.3f} mm"
                    )

                    print(
                        f"Minimum sigma_min "
                        f"扫描最小雅可比奇异值: "
                        f"{np.min(sigma_mins_scan):.6f}"
                    )

                    print(
                        f"Maximum condition diagnostic "
                        f"扫描最大条件数诊断值: "
                        f"{np.max(conditions_scan):.2f}"
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
                    "PASS if force/pose errors are small and saturation is zero "
                    "若力/位姿误差较小且无饱和则通过"
                )

                print(
                    "Close viewer to exit 关闭窗口结束"
                )

            viewer.sync()

            remaining = (
                dt
                - (
                    time.time()
                    - step_wall_start
                )
            )

            if remaining > 0.0:
                time.sleep(
                    remaining
                )


if __name__ == "__main__":
    main()
