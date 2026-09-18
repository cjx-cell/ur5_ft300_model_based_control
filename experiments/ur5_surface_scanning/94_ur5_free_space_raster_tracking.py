#!/usr/bin/env python3
"""94 - UR5 full-raster free-space dynamic tracking.

Purpose
-------
Step 93 proved that the same 961 scan poses are kinematically feasible for UR5.
Step 94 now isolates DYNAMICS from CONTACT:

- the robot follows the exact same solved UR5 joint path;
- the workpiece is shifted downward by 30 mm only in this verification scene;
- therefore the inspection tip follows the original scan surface in free space;
- control uses Pinocchio model-based computed torque;
- MuJoCo passive generalized forces are subtracted from the actuator command.

The raster trajectory is executed segment-by-segment.  The robot stops smoothly
at every row/connector transition so the sharp raster corners do not create
unphysical velocity jumps.

Outputs
-------
- ur5_ft300_inspection_free_space.xml
- terminal metrics for joint, TCP, orientation, torque and saturation

Close the MuJoCo viewer AFTER the final result has been printed.
"""

from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


HERE = Path(__file__).resolve().parent

SOURCE_WORKCELL = HERE / "ur5_ft300_inspection_workcell.xml"
FREE_SPACE_SCENE = HERE / "ur5_ft300_inspection_free_space.xml"

URDF_PATH = HERE / "ur5_ft300_inspection.urdf"
IK_PATH = HERE / "93_ur5_scan_ik_solutions.npz"
GEOMETRY_PATH = HERE / "89_raster_scan_path.npz"

FRAME_NAME = "inspection_tip"

FREE_SPACE_DROP = 0.030  # move workpiece down by 30 mm

# Average path speed inside each independently time-scaled segment.
SCAN_AVG_SPEED = 0.050       # m/s
CONNECTOR_AVG_SPEED = 0.030  # m/s
MIN_SEGMENT_DURATION = 0.35  # s

START_HOLD = 1.0
END_HOLD = 1.0

PRINT_INTERVAL = 1.0

# Simulation-only model-based tracking gains.
KP = np.array([64.0, 64.0, 64.0, 49.0, 36.0, 25.0])
KD = 2.0 * np.sqrt(KP)

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def build_free_space_scene():
    """Copy the workcell and lower only the workpiece by 30 mm."""
    tree = ET.parse(SOURCE_WORKCELL)
    root = tree.getroot()
    root.set("model", "ur5_ft300_inspection_free_space")

    geom = root.find(".//geom[@name='inspection_workpiece']")
    if geom is None:
        raise RuntimeError("找不到 inspection_workpiece")

    pos = np.fromstring(geom.attrib["pos"], sep=" ")
    pos[2] -= FREE_SPACE_DROP
    geom.set("pos", " ".join(f"{v:.9f}" for v in pos))

    root.insert(
        0,
        ET.Comment(
            " Free-space dynamics test: only the workpiece is lowered by "
            f"{FREE_SPACE_DROP * 1000:.0f} mm. "
            "Robot trajectory is unchanged. "
        ),
    )

    ET.indent(tree, space="  ")
    tree.write(
        FREE_SPACE_SCENE,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Catch XML/schema errors immediately.
    mujoco.MjModel.from_xml_path(str(FREE_SPACE_SCENE))


def sync_pinocchio_numerical_terms(pin_model, mj_model):
    """Copy MuJoCo armature/damping/friction metadata into Pinocchio."""
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


def contiguous_segments(row_id, phase):
    """Return overlapping path segments: scan row / connector / scan row / ..."""
    keys = list(zip(row_id.tolist(), phase.tolist()))

    runs = []
    run_start = 0

    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            runs.append((run_start, i - 1, keys[i - 1]))
            run_start = i

    runs.append((run_start, len(keys) - 1, keys[-1]))

    segments = []

    for run_index, (start, end, key) in enumerate(runs):
        if run_index > 0:
            start -= 1  # share the previous endpoint for exact continuity

        segments.append(
            {
                "start": start,
                "end": end,
                "row": int(key[0]),
                "phase": int(key[1]),
            }
        )

    return segments


def quintic_profile(u):
    """sigma, d sigma/du, d² sigma/du², u in [0,1]."""
    u = float(np.clip(u, 0.0, 1.0))

    sigma = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    dsigma = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    d2sigma = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

    return sigma, dsigma, d2sigma


def interp_vector(x, xp, values):
    out = np.empty(values.shape[1])

    for axis in range(values.shape[1]):
        out[axis] = np.interp(
            x,
            xp,
            values[:, axis],
        )

    return out


def make_segment_trajectory(segment, s_all, q_all):
    start = segment["start"]
    end = segment["end"]

    s = s_all[start : end + 1].copy()
    q = q_all[start : end + 1].copy()

    s_local = s - s[0]
    length = float(s_local[-1])

    if length <= 0.0:
        raise RuntimeError("轨迹段长度非正")

    edge_order = 2 if len(s_local) >= 3 else 1

    dq_ds = np.gradient(
        q,
        s_local,
        axis=0,
        edge_order=edge_order,
    )

    d2q_ds2 = np.gradient(
        dq_ds,
        s_local,
        axis=0,
        edge_order=edge_order,
    )

    speed = (
        SCAN_AVG_SPEED
        if segment["phase"] == 0
        else CONNECTOR_AVG_SPEED
    )

    duration = max(
        length / speed,
        MIN_SEGMENT_DURATION,
    )

    return {
        **segment,
        "s": s,
        "s_local": s_local,
        "q": q,
        "dq_ds": dq_ds,
        "d2q_ds2": d2q_ds2,
        "length": length,
        "duration": duration,
    }


def reference_from_segment(segment, local_time):
    T = segment["duration"]

    u = np.clip(
        local_time / T,
        0.0,
        1.0,
    )

    sigma, dsigma, d2sigma = quintic_profile(u)

    l = segment["length"]

    s_local = l * sigma
    sdot = l * dsigma / T
    sddot = l * d2sigma / (T * T)

    q_ref = interp_vector(
        s_local,
        segment["s_local"],
        segment["q"],
    )

    dq_ds = interp_vector(
        s_local,
        segment["s_local"],
        segment["dq_ds"],
    )

    d2q_ds2 = interp_vector(
        s_local,
        segment["s_local"],
        segment["d2q_ds2"],
    )

    v_ref = dq_ds * sdot

    a_ref = (
        d2q_ds2 * sdot * sdot
        + dq_ds * sddot
    )

    s_global = (
        segment["s"][0]
        + s_local
    )

    return (
        q_ref,
        v_ref,
        a_ref,
        s_global,
    )


def interp_pose(
    s_target,
    s_path,
    positions,
    rotations,
):
    if s_target <= s_path[0]:
        return (
            positions[0].copy(),
            rotations[0].copy(),
        )

    if s_target >= s_path[-1]:
        return (
            positions[-1].copy(),
            rotations[-1].copy(),
        )

    i = int(
        np.searchsorted(
            s_path,
            s_target,
        )
        - 1
    )

    i = max(
        0,
        min(
            i,
            len(s_path) - 2,
        ),
    )

    ds = (
        s_path[i + 1]
        - s_path[i]
    )

    alpha = (
        s_target - s_path[i]
    ) / ds

    p = (
        (1.0 - alpha)
        * positions[i]
        + alpha
        * positions[i + 1]
    )

    R0 = rotations[i]
    R1 = rotations[i + 1]

    delta = pin.log3(
        R0.T @ R1
    )

    R = (
        R0
        @ pin.exp3(
            alpha * delta
        )
    )

    return p, R


def orientation_error_angle(R_des, R):
    return float(
        np.linalg.norm(
            pin.log3(
                R_des @ R.T
            )
        )
    )


def main():
    for path in [
        SOURCE_WORKCELL,
        URDF_PATH,
        IK_PATH,
        GEOMETRY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    build_free_space_scene()

    ik_data = np.load(IK_PATH)
    geometry = np.load(GEOMETRY_PATH)

    q_path = ik_data["q"].copy()
    path_positions = geometry["position"]
    path_rotations = geometry["rotation"]
    s_path = geometry["arc_length"]
    row_id = geometry["row_id"]
    phase = geometry["phase"]

    if len(q_path) != len(s_path):
        raise RuntimeError(
            f"IK/path length mismatch: {len(q_path)} vs {len(s_path)}"
        )

    # All six UR5 joints are scalar revolute coordinates.  Unwrap across the
    # dense sequential IK path before differentiating.
    q_path = np.unwrap(
        q_path,
        axis=0,
    )

    segments_raw = contiguous_segments(
        row_id,
        phase,
    )

    segments = [
        make_segment_trajectory(
            segment,
            s_path,
            q_path,
        )
        for segment in segments_raw
    ]

    scan_duration = sum(
        segment["duration"]
        for segment in segments
    )

    total_motion_time = (
        START_HOLD
        + scan_duration
        + END_HOLD
    )

    mj_model = mujoco.MjModel.from_xml_path(
        str(FREE_SPACE_SCENE)
    )
    mj_data = mujoco.MjData(
        mj_model
    )

    pin_model = pin.buildModelFromUrdf(
        str(URDF_PATH)
    )
    pin_data = pin_model.createData()

    if (
        mj_model.nv != 6
        or pin_model.nv != 6
    ):
        raise RuntimeError(
            f"Unexpected robot dimensions: "
            f"MuJoCo nv={mj_model.nv}, "
            f"Pinocchio nv={pin_model.nv}"
        )

    sync_pinocchio_numerical_terms(
        pin_model,
        mj_model,
    )

    frame_id = pin_model.getFrameId(
        FRAME_NAME
    )
    if frame_id >= pin_model.nframes:
        raise RuntimeError(
            f"Pinocchio找不到frame: {FRAME_NAME}"
        )

    tip_site_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_SITE,
        "inspection_tip_site",
    )
    if tip_site_id < 0:
        raise RuntimeError(
            "MuJoCo找不到 inspection_tip_site"
        )

    # Start exactly at the first validated IK solution to isolate dynamic
    # tracking from approach planning.
    q_start_pin = q_path[0].copy()
    q_start_mj = pin_q_to_mj(
        q_start_pin,
        pin_model,
        mj_model,
    )

    mj_data.qpos[:] = q_start_mj
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    if mj_data.ncon != 0:
        raise RuntimeError(
            f"Free-space start should have zero contacts, "
            f"but ncon={mj_data.ncon}"
        )

    # Global metrics.
    max_joint_error = 0.0
    max_tcp_error = 0.0
    max_orientation_error = 0.0
    max_torque = 0.0
    saturation_steps = 0
    control_steps = 0

    segment_index = 0
    segment_elapsed = 0.0
    finished = False
    result_printed = False

    last_print_time = -PRINT_INTERVAL

    print(
        "========== 94 UR5 Free-Space Tracking "
        "94 UR5自由空间动态扫描 =========="
    )
    print(
        f"Path length 路径长度: "
        f"{s_path[-1] * 1000.0:.1f} mm"
    )
    print(
        f"Segments 轨迹段数: {len(segments)} "
        f"(7 scan rows + 6 connectors)"
    )
    print(
        f"Workpiece drop 自由空间工件下移: "
        f"{FREE_SPACE_DROP * 1000.0:.1f} mm"
    )
    print(
        f"Planned scan time 计划扫描时间: "
        f"{scan_duration:.1f} s"
    )
    print(
        "Controller 控制器: "
        "Pinocchio computed torque + MuJoCo passive-force compensation"
    )

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.00, -0.15, 0.82]
        )
        viewer.cam.distance = 2.05
        viewer.cam.azimuth = 142.0
        viewer.cam.elevation = -23.0

        wall_start = time.time()

        while viewer.is_running():
            step_wall_start = time.time()

            t = float(mj_data.time)

            # ------------------------------------------------------------
            # Reference generation
            # ------------------------------------------------------------
            if t < START_HOLD:
                q_ref = q_path[0].copy()
                v_ref = np.zeros(6)
                a_ref = np.zeros(6)
                s_ref = float(s_path[0])
                phase_text = "hold 保持"

            elif not finished:
                motion_time = t - START_HOLD

                # Advance through completed segments.
                while (
                    segment_index
                    < len(segments)
                    and motion_time
                    >= segment_elapsed
                    + segments[segment_index]["duration"]
                ):
                    segment_elapsed += (
                        segments[segment_index]["duration"]
                    )
                    segment_index += 1

                if segment_index >= len(segments):
                    finished = True

                    q_ref = q_path[-1].copy()
                    v_ref = np.zeros(6)
                    a_ref = np.zeros(6)
                    s_ref = float(s_path[-1])
                    phase_text = "finished 完成"

                else:
                    segment = segments[
                        segment_index
                    ]

                    local_time = (
                        motion_time
                        - segment_elapsed
                    )

                    (
                        q_ref,
                        v_ref,
                        a_ref,
                        s_ref,
                    ) = reference_from_segment(
                        segment,
                        local_time,
                    )

                    phase_text = (
                        "scan 扫描"
                        if segment["phase"] == 0
                        else "turn 换行"
                    )

            else:
                q_ref = q_path[-1].copy()
                v_ref = np.zeros(6)
                a_ref = np.zeros(6)
                s_ref = float(s_path[-1])
                phase_text = "finished 完成"

            # ------------------------------------------------------------
            # Current state -> Pinocchio
            # ------------------------------------------------------------
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

            # Configuration error in the robot tangent space.
            e_q = pin.difference(
                pin_model,
                q_pin,
                q_ref,
            )

            e_v = (
                v_ref
                - v_pin
            )

            ddq_cmd = (
                a_ref
                + KP * e_q
                + KD * e_v
            )

            M = pin.crba(
                pin_model,
                pin_data,
                q_pin,
            )
            M = 0.5 * (
                M + M.T
            )

            h = pin.nonLinearEffects(
                pin_model,
                pin_data,
                q_pin,
                v_pin,
            )

            tau_target_pin = (
                M @ ddq_cmd
                + h
            )

            tau_target_mj = pin_tau_to_mj(
                tau_target_pin,
                pin_model,
                mj_model,
            )

            # MuJoCo already applies qfrc_passive on the RHS.
            tau_act = (
                tau_target_mj
                - mj_data.qfrc_passive[:6]
            )

            saturated = False

            for i in range(6):
                if mj_model.actuator_ctrllimited[i]:
                    low, high = (
                        mj_model.actuator_ctrlrange[i]
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

            if saturated:
                saturation_steps += 1

            control_steps += 1

            mj_data.ctrl[:6] = tau_act

            mujoco.mj_step(
                mj_model,
                mj_data,
            )

            # ------------------------------------------------------------
            # Metrics after integration
            # ------------------------------------------------------------
            q_after = mj_q_to_pin(
                mj_data.qpos,
                mj_model,
                pin_model,
            )

            joint_error = float(
                np.linalg.norm(
                    pin.difference(
                        pin_model,
                        q_after,
                        q_ref,
                    )
                )
            )

            max_joint_error = max(
                max_joint_error,
                joint_error,
            )

            p_des, R_des = interp_pose(
                s_ref,
                s_path,
                path_positions,
                path_rotations,
            )

            p_actual = (
                mj_data.site_xpos[
                    tip_site_id
                ].copy()
            )

            R_actual = (
                mj_data.site_xmat[
                    tip_site_id
                ]
                .reshape(3, 3)
                .copy()
            )

            tcp_error = float(
                np.linalg.norm(
                    p_des
                    - p_actual
                )
            )

            orientation_error = (
                orientation_error_angle(
                    R_des,
                    R_actual,
                )
            )

            max_tcp_error = max(
                max_tcp_error,
                tcp_error,
            )

            max_orientation_error = max(
                max_orientation_error,
                orientation_error,
            )

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

            # No environment contact is expected in this step.
            if mj_data.ncon != 0:
                print(
                    f"Unexpected contact 非预期接触: "
                    f"time={mj_data.time:.3f} s, "
                    f"ncon={mj_data.ncon}"
                )
                raise RuntimeError(
                    "Free-space dynamic test encountered contact"
                )

            if (
                mj_data.time
                - last_print_time
                >= PRINT_INTERVAL
            ):
                last_print_time = (
                    mj_data.time
                )

                progress = (
                    100.0
                    * s_ref
                    / s_path[-1]
                )

                print(
                    f"Time 时间: {mj_data.time:5.1f} s | "
                    f"{phase_text} | "
                    f"progress={progress:5.1f}% | "
                    f"TCP={tcp_error * 1000.0:.3f} mm | "
                    f"ori={np.degrees(orientation_error):.3f} deg | "
                    f"tau_max={np.max(np.abs(tau_act)):.1f} Nm"
                )

            # ------------------------------------------------------------
            # Print final result BEFORE the viewer is closed.
            # ------------------------------------------------------------
            if (
                finished
                and not result_printed
                and t
                >= START_HOLD
                + scan_duration
                + END_HOLD
            ):
                result_printed = True

                saturation_ratio = (
                    100.0
                    * saturation_steps
                    / max(
                        control_steps,
                        1,
                    )
                )

                print(
                    "\n========== 94 Result 94结果 =========="
                )
                print(
                    f"Max joint tracking error "
                    f"最大关节跟踪误差: "
                    f"{np.degrees(max_joint_error):.4f} deg"
                )
                print(
                    f"Max TCP error 最大TCP误差: "
                    f"{max_tcp_error * 1000.0:.4f} mm"
                )
                print(
                    f"Max orientation error 最大姿态误差: "
                    f"{np.degrees(max_orientation_error):.4f} deg"
                )
                print(
                    f"Max actuator torque 最大执行器力矩: "
                    f"{max_torque:.3f} Nm"
                )
                print(
                    f"Torque saturation 力矩饱和步占比: "
                    f"{saturation_ratio:.3f} %"
                )
                print(
                    f"Unexpected contacts 非预期接触: 0"
                )
                print(
                    "Result 结果: PASS if tracking errors and saturation "
                    "are acceptable; keep viewer open for visual inspection. "
                    "若跟踪误差与饱和比例合理则通过；可继续观察窗口。"
                )
                print(
                    "Close viewer to exit 关闭窗口结束"
                )

            viewer.sync()

            # Rough real-time pacing for visualization.
            remaining = (
                mj_model.opt.timestep
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
