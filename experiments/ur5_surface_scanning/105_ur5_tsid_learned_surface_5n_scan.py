#!/usr/bin/env python3
"""105 - Full learned-surface 5 N scan using TSID/HQP and FT300 feedback.

The desired position, normal and orientation come only from the 97B learned
surface.  MuJoCo computes physical contact; FT300 force and TCP-shifted moment
adapt normal position and roll/pitch online.  TSID solves Cartesian inverse
dynamics with joint and actual motor-command bounds inside the HQP.
"""

import argparse
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", message=".*to-Python converter.*already registered.*")

import mujoco
import numpy as np
import pinocchio as pin
import tsid

from ur5_tsid_mujoco_common import (
    actuator_bounds,
    inverse_dynamics_residual,
    load_robot_system,
    read_state,
    reset_mujoco,
    se3_sample,
    tau_to_ctrl,
)


HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "ur5_ft300_inspection_workcell.xml"
URDF_PATH = HERE / "ur5_ft300_inspection.urdf"
GEOMETRY_PATH = HERE / "97B_reconstructed_surface.npz"
IK_PATH = HERE / "97C_learned_surface_ik_solutions.npz"
BASELINE_PATH = HERE / "97D_ur5_learned_surface_5n_scan.py"
OUTPUT_PATH = HERE / "105_tsid_learned_surface_5n_scan_result.npz"

FRAME_NAME = "inspection_tip"
SENSOR_FRAME_NAME = "robotiq_ft_frame_id"

PRECONTACT_GAP = 0.020
MAX_APPROACH_PENETRATION = 0.002
APPROACH_DURATION = 6.0
BIAS_START = 0.20
BIAS_END = 0.90
CONTACT_THRESHOLD = 0.25

DESIRED_FORCE = 5.0
FORCE_RAMP_DURATION = 2.0
PRE_SCAN_FORCE_HOLD = 1.0
POST_SCAN_FORCE_HOLD = 1.0
FORCE_MASS = 2.0
FORCE_DAMPING = 120.0
NORMAL_VELOCITY_LIMIT = 0.004
NORMAL_ACCEL_LIMIT = 0.15
NORMAL_CORRECTION_MIN = -0.002
NORMAL_CORRECTION_MAX = 0.006
FILTER_CUTOFF_HZ = 20.0

TILT_INERTIA = 0.08
TILT_DAMPING = 1.20
TILT_STIFFNESS = 8.0
TILT_ANGLE_LIMIT = np.deg2rad(5.0)
TILT_VELOCITY_LIMIT = np.deg2rad(20.0)
TILT_ACCELERATION_LIMIT = np.deg2rad(100.0)

KP_SE3 = np.array([225.0, 225.0, 225.0, 144.0, 144.0, 144.0])
KD_SE3 = 2.0 * np.sqrt(KP_SE3)
JOINT_VELOCITY_LIMIT = 3.0
JOINT_ACCELERATION_LIMIT = 80.0

PASS_FORCE_MEAN_ERROR = 0.50
PASS_FORCE_MAX_ERROR = 3.00
PASS_TCP_MAX_ERROR = 0.002
PASS_NORMAL_MAX_ERROR_DEG = 1.00
PRINT_INTERVAL = 1.0


def load_baseline_helpers():
    """Reuse the exact validated 97D path interpolation and contact checks."""
    spec = importlib.util.spec_from_file_location("baseline_97d", BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def object_id(model, object_type, name):
    index = mujoco.mj_name2id(model, object_type, name)
    if index < 0:
        raise RuntimeError(f"MuJoCo找不到对象: {name}")
    return index


def is_descendant(model, body_id, root_body_id):
    current = int(body_id)
    while current > 0:
        if current == root_body_id:
            return True
        current = int(model.body_parentid[current])
    return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="打开MuJoCo Viewer")
    return parser.parse_args(argv)


def main(argv=None, raise_on_failure=True):
    args = parse_args(argv)
    base = load_baseline_helpers()
    geometry = np.load(GEOMETRY_PATH)
    ik_data = np.load(IK_PATH)
    base.validate_learned_path(geometry, ik_data)

    positions = geometry["position"].copy()
    normals = geometry["surface_normal"].copy()
    rotations = geometry["rotation"].copy()
    arc_length = geometry["arc_length"].copy()
    row_id = geometry["row_id"].copy()
    path_phase = geometry["phase"].copy()
    raw_segments = base.contiguous_segments(row_id, path_phase)
    segments = [
        base.make_segment(
            raw, arc_length, positions, normals, rotations
        )
        for raw in raw_segments
    ]
    scan_duration = float(sum(segment["duration"] for segment in segments))

    p_start = positions[0].copy()
    n_start = base.normalize(normals[0])
    rotation_start = rotations[0].copy()
    if np.dot(rotation_start[:, 2], -n_start) < 1.0 - 1e-10:
        raise RuntimeError("初始工具轴与学习法向不一致")

    system = load_robot_system(MJCF_PATH, URDF_PATH)
    mj_model, mj_data = system.mj_model, system.mj_data
    model, robot = system.pin_model, system.robot
    frame_id = model.getFrameId(FRAME_NAME)
    sensor_frame_id = model.getFrameId(SENSOR_FRAME_NAME)
    if frame_id >= model.nframes or sensor_frame_id >= model.nframes:
        raise RuntimeError("Pinocchio缺少TCP或FT300 frame")

    ft_site_id = object_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "ft300_site")
    tip_site_id = object_id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip_site"
    )
    force_id = object_id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft300_force")
    torque_id = object_id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft300_torque")
    force_adr = int(mj_model.sensor_adr[force_id])
    torque_adr = int(mj_model.sensor_adr[torque_id])

    sensor_body_id = int(mj_model.site_bodyid[ft_site_id])
    payload_body_ids = [
        body_id
        for body_id in range(mj_model.nbody)
        if is_descendant(mj_model, body_id, sensor_body_id)
    ]
    payload_mass = float(sum(mj_model.body_mass[i] for i in payload_body_ids))

    p_precontact = p_start + PRECONTACT_GAP * n_start
    ik_work = model.createData()
    q_precontact = base.solve_pose_ik(
        model,
        ik_work,
        frame_id,
        ik_data["q"][0].copy(),
        p_precontact,
        rotation_start,
    )
    reset_mujoco(system, q_precontact)
    if mj_data.ncon:
        raise RuntimeError(f"预接触位姿存在碰撞: ncon={mj_data.ncon}")

    def read_raw_wrench():
        return np.concatenate(
            (
                mj_data.sensordata[force_adr : force_adr + 3].copy(),
                mj_data.sensordata[torque_adr : torque_adr + 3].copy(),
            )
        )

    def payload_gravity_wrench():
        weighted_com = sum(
            (mj_model.body_mass[i] * mj_data.xipos[i] for i in payload_body_ids),
            start=np.zeros(3),
        )
        payload_com = weighted_com / payload_mass
        sensor_position = mj_data.site_xpos[ft_site_id]
        force_world = -payload_mass * np.asarray(mj_model.opt.gravity)
        moment_world = np.cross(payload_com - sensor_position, force_world)
        sensor_rotation = mj_data.site_xmat[ft_site_id].reshape(3, 3)
        return np.concatenate(
            (sensor_rotation.T @ force_world, sensor_rotation.T @ moment_world)
        )

    def external_world_wrench(wrench_sensor):
        sensor_rotation = mj_data.site_xmat[ft_site_id].reshape(3, 3)
        return np.concatenate(
            (
                -sensor_rotation @ wrench_sensor[:3],
                -sensor_rotation @ wrench_sensor[3:],
            )
        )

    formulation = tsid.InverseDynamicsFormulationAccForce(
        "ur5-learned-scan", robot, False
    )
    formulation.computeProblemData(0.0, q_precontact, np.zeros(model.nv))

    se3_task = tsid.TaskSE3Equality("learned-tcp-pose", robot, FRAME_NAME)
    se3_task.useLocalFrame(False)
    se3_task.setMask(np.ones(6))
    se3_task.setKp(KP_SE3)
    se3_task.setKd(KD_SE3)
    se3_task.setReference(se3_sample(pin.SE3(rotation_start, p_precontact)))

    dt = float(mj_model.opt.timestep)
    joint_bounds = tsid.TaskJointPosVelAccBounds(
        "joint-safety", robot, dt, False
    )
    joint_bounds.setMask(np.ones(robot.na))
    joint_bounds.setPositionBounds(
        model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    )
    joint_bounds.setVelocityBounds(JOINT_VELOCITY_LIMIT * np.ones(robot.na))
    joint_bounds.setAccelerationBounds(
        JOINT_ACCELERATION_LIMIT * np.ones(robot.na)
    )
    joint_bounds.setImposeBounds(True, True, True, True)

    motor_lower, motor_upper = actuator_bounds(system)
    motor_bounds = tsid.TaskActuationBounds("motor-safety", robot)
    motor_bounds.setMask(np.ones(robot.na))
    motor_bounds.setBounds(motor_lower, motor_upper)
    formulation.addMotionTask(joint_bounds, 1.0, 0, 0.0)
    formulation.addActuationTask(motor_bounds, 1.0, 0, 0.0)
    formulation.addMotionTask(se3_task, 1.0, 1, 0.0)

    solver = tsid.SolverHQuadProgFast("ur5-learned-scan-qpoases")
    solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

    filter_time_constant = 1.0 / (2.0 * np.pi * FILTER_CUTOFF_HZ)
    filter_alpha = dt / (filter_time_constant + dt)
    filtered_wrench = np.zeros(6)
    bias_sum = np.zeros(6)
    bias_count = 0
    bias = None
    contact_time = None
    force_sign = None
    scan_start_time = None
    scan_finish_time = None
    segment_index = 0
    segment_elapsed = 0.0
    normal_correction = 0.0
    normal_velocity = 0.0
    tilt_correction = np.zeros(2)
    tilt_velocity = np.zeros(2)

    metrics = {
        "time": [],
        "progress": [],
        "force_desired": [],
        "force_measured": [],
        "force_error": [],
        "tcp_error": [],
        "normal_error_deg": [],
        "learned_normal_deviation_deg": [],
        "tangential_force": [],
        "tilt_moment_norm": [],
        "tilt_angle_deg": [],
        "normal_correction": [],
        "sigma_min": [],
        "condition": [],
        "desired_position": [],
        "actual_position": [],
        "actual_tool_z": [],
        "actuator_torque": [],
        "qpos": [],
    }
    max_ctrl = 0.0
    max_motor_violation = 0.0
    max_dynamics_residual = 0.0
    qp_failures = 0
    unexpected_contact_count = 0
    solve_times = []
    dynamics_data = model.createData()
    kinematics_data = model.createData()
    completed = False
    last_print = -1.0

    print("========== 105 TSID/HQP Learned-Surface 5 N Scan ==========")
    print(f"Path poses/path length 路径位姿/长度: {len(positions)}/{arc_length[-1] * 1000:.1f} mm")
    print(f"Segments/planned time 轨迹段/计划时间: {len(segments)}/{scan_duration:.1f} s")
    print("Desired geometry 期望几何: 97B learned surface only")
    print("Controller 控制器: FT300 admittance + TSID/HQP hard bounds")
    print("Post-solve clipping 求解后裁剪: DISABLED")

    if args.viewer:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(mj_model, mj_data)
    else:
        context = nullcontext(None)

    maximum_time = (
        BIAS_END
        + APPROACH_DURATION
        + FORCE_RAMP_DURATION
        + PRE_SCAN_FORCE_HOLD
        + scan_duration
        + POST_SCAN_FORCE_HOLD
        + 2.0
    )

    with context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = np.array([0.02, -0.18, 0.82])
            viewer.cam.distance = 1.85
            viewer.cam.azimuth = 142.0
            viewer.cam.elevation = -24.0

        while mj_data.time < maximum_time:
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.perf_counter()
            t = float(mj_data.time)

            raw = read_raw_wrench()
            gravity = payload_gravity_wrench()
            if BIAS_START <= t < BIAS_END:
                bias_sum += raw - gravity
                bias_count += 1
            if bias is None and t >= BIAS_END:
                if bias_count == 0:
                    raise RuntimeError("没有FT300零偏样本")
                bias = bias_sum / bias_count
                print("Bias calibration FT300零偏标定: completed 完成")
            compensated = np.zeros(6) if bias is None else raw - gravity - bias
            filtered_wrench += filter_alpha * (compensated - filtered_wrench)
            wrench_world = external_world_wrench(filtered_wrench)

            desired_force = 0.0
            nominal_position = p_start.copy()
            current_normal = n_start.copy()
            nominal_rotation = rotation_start.copy()
            nominal_velocity = np.zeros(3)
            nominal_acceleration = np.zeros(3)
            desired_omega = np.zeros(3)
            desired_alpha = np.zeros(3)
            n_s = np.zeros(3)
            n_ss = np.zeros(3)
            sdot = 0.0
            sddot = 0.0
            progress = 0.0

            if t < BIAS_END:
                phase_text = "bias/hold 零偏标定"
                desired_position = p_precontact.copy()
                desired_velocity = np.zeros(3)
                desired_acceleration = np.zeros(3)
            elif contact_time is None:
                phase_text = "approach 接近"
                s, ds, d2s = base.quintic((t - BIAS_END) / APPROACH_DURATION)
                delta = -MAX_APPROACH_PENETRATION - PRECONTACT_GAP
                offset = PRECONTACT_GAP + s * delta
                desired_position = p_start + offset * n_start
                desired_velocity = ds * delta / APPROACH_DURATION * n_start
                desired_acceleration = (
                    d2s * delta / APPROACH_DURATION**2 * n_start
                )
            else:
                elapsed_contact = t - contact_time
                if elapsed_contact < FORCE_RAMP_DURATION:
                    phase_text = "force ramp 力建立"
                    desired_force = DESIRED_FORCE * base.quintic(
                        elapsed_contact / FORCE_RAMP_DURATION
                    )[0]
                elif elapsed_contact < FORCE_RAMP_DURATION + PRE_SCAN_FORCE_HOLD:
                    phase_text = "force hold 恒力预保持"
                    desired_force = DESIRED_FORCE
                else:
                    desired_force = DESIRED_FORCE
                    if scan_start_time is None:
                        scan_start_time = t
                        print(f"Scan started 扫描开始: t={t:.3f}s")
                    scan_elapsed = t - scan_start_time
                    while (
                        segment_index < len(segments)
                        and scan_elapsed
                        >= segment_elapsed + segments[segment_index]["duration"]
                    ):
                        segment_elapsed += segments[segment_index]["duration"]
                        segment_index += 1
                    if segment_index < len(segments):
                        segment = segments[segment_index]
                        reference = base.segment_reference(
                            segment, scan_elapsed - segment_elapsed
                        )
                        (
                            nominal_position,
                            current_normal,
                            nominal_rotation,
                            p_s,
                            p_ss,
                            n_s,
                            n_ss,
                            omega_s,
                            omega_ss,
                            sdot,
                            sddot,
                            s_global,
                        ) = reference
                        nominal_velocity = p_s * sdot
                        nominal_acceleration = p_ss * sdot**2 + p_s * sddot
                        desired_omega = omega_s * sdot
                        desired_alpha = omega_ss * sdot**2 + omega_s * sddot
                        progress = 100.0 * s_global / arc_length[-1]
                        phase_text = "scan 扫描" if segment["phase"] == 0 else "turn 换行"
                    else:
                        if scan_finish_time is None:
                            scan_finish_time = t
                            print(f"Scan finished 扫描完成: t={t:.3f}s")
                        phase_text = "post hold 结束保持"
                        nominal_position = positions[-1].copy()
                        current_normal = base.normalize(normals[-1])
                        nominal_rotation = rotations[-1].copy()
                        progress = 100.0

            projected_force = float(np.dot(wrench_world[:3], current_normal))
            measured_force = (
                abs(projected_force)
                if force_sign is None
                else max(0.0, force_sign * projected_force)
            )
            if (
                contact_time is None
                and bias is not None
                and t >= BIAS_END
                and measured_force >= CONTACT_THRESHOLD
            ):
                contact_time = t
                force_sign = 1.0 if projected_force >= 0.0 else -1.0
                normal_correction = float(
                    np.clip(
                        -np.dot(mj_data.site_xpos[tip_site_id] - p_start, n_start),
                        NORMAL_CORRECTION_MIN,
                        NORMAL_CORRECTION_MAX,
                    )
                )
                normal_velocity = 0.0
                print(f"Contact detected 检测到接触: t={t:.3f}s F={measured_force:.3f}N")

            normal_acceleration = 0.0
            if contact_time is not None:
                normal_acceleration = float(
                    np.clip(
                        (desired_force - measured_force - FORCE_DAMPING * normal_velocity)
                        / FORCE_MASS,
                        -NORMAL_ACCEL_LIMIT,
                        NORMAL_ACCEL_LIMIT,
                    )
                )
                normal_velocity = float(
                    np.clip(
                        normal_velocity + normal_acceleration * dt,
                        -NORMAL_VELOCITY_LIMIT,
                        NORMAL_VELOCITY_LIMIT,
                    )
                )
                normal_correction = float(
                    np.clip(
                        normal_correction + normal_velocity * dt,
                        NORMAL_CORRECTION_MIN,
                        NORMAL_CORRECTION_MAX,
                    )
                )
                desired_position = nominal_position - normal_correction * current_normal
                if phase_text in {"scan 扫描", "turn 换行"}:
                    desired_velocity = (
                        nominal_velocity
                        - normal_velocity * current_normal
                        - normal_correction * n_s * sdot
                    )
                    desired_acceleration = (
                        nominal_acceleration
                        - normal_acceleration * current_normal
                        - 2.0 * normal_velocity * n_s * sdot
                        - normal_correction * (n_ss * sdot**2 + n_s * sddot)
                    )
                else:
                    desired_velocity = -normal_velocity * current_normal
                    desired_acceleration = -normal_acceleration * current_normal

            sensor_moment = wrench_world[3:].copy()
            sensor_to_tip = (
                mj_data.site_xpos[tip_site_id] - mj_data.site_xpos[ft_site_id]
            )
            moment_tcp = sensor_moment - np.cross(sensor_to_tip, wrench_world[:3])
            tilt_moment = (nominal_rotation.T @ moment_tcp)[:2]
            tilt_acceleration = np.zeros(2)
            if contact_time is not None:
                tilt_acceleration = base.clip_vector_norm(
                    (
                        -tilt_moment
                        - TILT_DAMPING * tilt_velocity
                        - TILT_STIFFNESS * tilt_correction
                    )
                    / TILT_INERTIA,
                    TILT_ACCELERATION_LIMIT,
                )
                tilt_velocity = base.clip_vector_norm(
                    tilt_velocity + tilt_acceleration * dt,
                    TILT_VELOCITY_LIMIT,
                )
                proposed = tilt_correction + tilt_velocity * dt
                tilt_correction = base.clip_vector_norm(proposed, TILT_ANGLE_LIMIT)

            correction = np.array([tilt_correction[0], tilt_correction[1], 0.0])
            correction_velocity = nominal_rotation @ np.array(
                [tilt_velocity[0], tilt_velocity[1], 0.0]
            )
            correction_acceleration = (
                nominal_rotation
                @ np.array([tilt_acceleration[0], tilt_acceleration[1], 0.0])
                + np.cross(desired_omega, correction_velocity)
            )
            desired_rotation = nominal_rotation @ pin.exp3(correction)
            desired_omega = desired_omega + correction_velocity
            desired_alpha = desired_alpha + correction_acceleration

            sample = se3_sample(pin.SE3(desired_rotation, desired_position))
            sample.derivative(np.concatenate((desired_velocity, desired_omega)))
            sample.second_derivative(
                np.concatenate((desired_acceleration, desired_alpha))
            )
            se3_task.setReference(sample)

            q, v = read_state(system)
            pin.computeJointJacobians(model, kinematics_data, q)
            pin.updateFramePlacements(model, kinematics_data)
            sensor_jacobian = pin.getFrameJacobian(
                model,
                kinematics_data,
                sensor_frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            tcp_jacobian = pin.getFrameJacobian(
                model, kinematics_data, frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            singular_values = np.linalg.svd(tcp_jacobian, compute_uv=False)
            sigma_min = float(singular_values[-1])
            condition = float(singular_values[0] / singular_values[-1])
            external_tau = sensor_jacobian.T @ wrench_world
            passive = np.zeros(robot.na)
            for item in system.joints:
                passive[item.pin_v] = mj_data.qfrc_passive[item.mj_v]
            motor_bounds.setBounds(
                motor_lower + external_tau + passive,
                motor_upper + external_tau + passive,
            )

            problem = formulation.computeProblemData(t, q, v)
            solve_start = time.perf_counter()
            solution = solver.solve(problem)
            solve_times.append((time.perf_counter() - solve_start) * 1e6)
            if solution.status != 0:
                qp_failures += 1
                print(f"QP failed QP失败: t={t:.3f}s status={solution.status}")
                break
            tau_tsid = np.asarray(formulation.getActuatorForces(solution)).reshape(-1)
            ddq = np.asarray(formulation.getAccelerations(solution)).reshape(-1)
            max_dynamics_residual = max(
                max_dynamics_residual,
                inverse_dynamics_residual(
                    system, dynamics_data, q, v, ddq, tau_tsid
                ),
            )
            ctrl, violation = tau_to_ctrl(
                system, tau_tsid - external_tau, clip=False
            )
            max_motor_violation = max(max_motor_violation, violation)
            max_ctrl = max(max_ctrl, float(np.max(np.abs(ctrl))))
            mj_data.ctrl[:] = ctrl
            mujoco.mj_step(mj_model, mj_data)

            bad_contacts = base.unexpected_contacts(mj_model, mj_data)
            if bad_contacts:
                unexpected_contact_count += 1
                raise RuntimeError(f"非预期碰撞: {bad_contacts}")

            actual_position = mj_data.site_xpos[tip_site_id].copy()
            actual_rotation = mj_data.site_xmat[tip_site_id].reshape(3, 3).copy()
            tcp_error = float(np.linalg.norm(desired_position - actual_position))
            normal_error_deg = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            np.dot(actual_rotation[:, 2], desired_rotation[:, 2]),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            learned_deviation_deg = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(np.dot(actual_rotation[:, 2], -current_normal), -1.0, 1.0)
                    )
                )
            )
            tangential = wrench_world[:3] - np.dot(
                wrench_world[:3], current_normal
            ) * current_normal

            if scan_start_time is not None and scan_finish_time is None:
                metrics["time"].append(t - scan_start_time)
                metrics["progress"].append(progress)
                metrics["force_desired"].append(desired_force)
                metrics["force_measured"].append(measured_force)
                metrics["force_error"].append(abs(DESIRED_FORCE - measured_force))
                metrics["tcp_error"].append(tcp_error)
                metrics["normal_error_deg"].append(normal_error_deg)
                metrics["learned_normal_deviation_deg"].append(learned_deviation_deg)
                metrics["tangential_force"].append(np.linalg.norm(tangential))
                metrics["tilt_moment_norm"].append(np.linalg.norm(tilt_moment))
                metrics["tilt_angle_deg"].append(np.rad2deg(np.linalg.norm(tilt_correction)))
                metrics["normal_correction"].append(normal_correction)
                metrics["sigma_min"].append(sigma_min)
                metrics["condition"].append(condition)
                metrics["desired_position"].append(desired_position.copy())
                metrics["actual_position"].append(actual_position)
                metrics["actual_tool_z"].append(actual_rotation[:, 2].copy())
                metrics["actuator_torque"].append(ctrl.copy())
                metrics["qpos"].append(mj_data.qpos.copy())

            if t - last_print >= PRINT_INTERVAL:
                last_print = t
                print(
                    f"t={t:5.1f}s | {phase_text:18s} | progress={progress:5.1f}% | "
                    f"F={desired_force:4.2f}/{measured_force:4.2f}N | "
                    f"TCP={tcp_error * 1000:.3f}mm | tilt={np.rad2deg(np.linalg.norm(tilt_correction)):.3f}deg | "
                    f"tau={np.max(np.abs(ctrl)):.1f}Nm"
                )

            if contact_time is None and t > BIAS_END + APPROACH_DURATION + 0.25:
                raise RuntimeError("接近结束仍未检测到接触")
            if (
                scan_finish_time is not None
                and t >= scan_finish_time + POST_SCAN_FORCE_HOLD
            ):
                completed = True

            if viewer is not None:
                viewer.sync()
                remaining = dt - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            if completed:
                break

    force_error = np.asarray(metrics["force_error"])
    tcp_error = np.asarray(metrics["tcp_error"])
    normal_error = np.asarray(metrics["normal_error_deg"])
    solve_times = np.asarray(solve_times)
    result_pass = bool(
        completed
        and force_error.size
        and qp_failures == 0
        and unexpected_contact_count == 0
        and max_motor_violation < 1e-7
        and np.mean(force_error) <= PASS_FORCE_MEAN_ERROR
        and np.max(force_error) <= PASS_FORCE_MAX_ERROR
        and np.max(tcp_error) <= PASS_TCP_MAX_ERROR
        and np.max(normal_error) <= PASS_NORMAL_MAX_ERROR_DEG
        and max_dynamics_residual < 1e-8
    )

    save_data = {
        "source_97b": np.asarray(GEOMETRY_PATH.name),
        "source_97c": np.asarray(IK_PATH.name),
        "controller": np.asarray("TSID/HQP + FT300 admittance"),
        "analytic_surface_used": np.asarray(False),
        "result_pass": np.asarray(result_pass),
        "target_force": np.asarray(DESIRED_FORCE),
        "max_actuator_torque": np.asarray(max_ctrl),
        "max_motor_bound_violation": np.asarray(max_motor_violation),
        "learned_position": positions,
        "learned_surface_normal": normals,
        "learned_rotation": rotations,
        "learned_arc_length": arc_length,
        "row_id": row_id,
        "phase": path_phase,
    }
    save_data.update({key: np.asarray(value) for key, value in metrics.items()})
    np.savez_compressed(OUTPUT_PATH, **save_data)

    scan_actual_duration = (
        np.nan
        if scan_start_time is None or scan_finish_time is None
        else scan_finish_time - scan_start_time
    )
    print("\n========== 105 Result 结果 ==========")
    print(f"Scan duration 实际扫描时间: {scan_actual_duration:.3f} s")
    print(f"Force mean/max error 力平均/最大误差: {np.mean(force_error):.3f}/{np.max(force_error):.3f} N")
    print(f"TCP mean/max error: {np.mean(tcp_error) * 1000:.3f}/{np.max(tcp_error) * 1000:.3f} mm")
    print(f"Orientation mean/max error: {np.mean(normal_error):.3f}/{np.max(normal_error):.3f} deg")
    print(f"Learned-normal mean/max correction: {np.mean(metrics['learned_normal_deviation_deg']):.3f}/{np.max(metrics['learned_normal_deviation_deg']):.3f} deg")
    print(f"Normal correction min/max: {np.min(metrics['normal_correction']) * 1000:+.3f}/{np.max(metrics['normal_correction']) * 1000:+.3f} mm")
    print(f"Minimum sigma / maximum condition: {np.min(metrics['sigma_min']):.6f}/{np.max(metrics['condition']):.2f}")
    print(f"Maximum actuator torque 最大电机力矩: {max_ctrl:.3f} Nm")
    print(f"Motor-bound violation 电机约束违规: {max_motor_violation:.3e} Nm")
    print(f"QP failures/unexpected contacts: {qp_failures}/{unexpected_contact_count}")
    print(f"Dynamics residual 动力学残差: {max_dynamics_residual:.3e}")
    if solve_times.size:
        print(f"QP solve mean/P99: {np.mean(solve_times):.1f}/{np.percentile(solve_times, 99):.1f} us")
    print(f"Saved 保存: {OUTPUT_PATH.name}")
    print(f"Result 结果: {'PASS 通过' if result_pass else 'FAIL 失败'}")
    if not result_pass and raise_on_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
