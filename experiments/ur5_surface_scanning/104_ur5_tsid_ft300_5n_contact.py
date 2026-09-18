#!/usr/bin/env python3
"""104 - TSID/HQP contact state machine with FT300 5 N force control.

MuJoCo owns contact physics and sensor generation.  TSID owns Cartesian motion,
inverse dynamics and hard safety bounds.  The FT300 signal closes an outer
normal-admittance loop and its measured wrench is compensated in joint torque.

Phases: bias hold -> approach -> force ramp -> 5 N hold -> withdraw.
"""

import argparse
from contextlib import nullcontext
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
GEOMETRY_PATH = HERE / "89_raster_scan_path.npz"
IK_PATH = HERE / "93_ur5_scan_ik_solutions.npz"
OUTPUT_PATH = HERE / "104_tsid_ft300_contact_result.npz"

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
FORCE_HOLD_DURATION = 5.0
WITHDRAW_GAP = 0.015
WITHDRAW_DURATION = 2.0

FORCE_MASS = 2.0
FORCE_DAMPING = 120.0
NORMAL_VELOCITY_LIMIT = 0.004
NORMAL_ACCEL_LIMIT = 0.15
NORMAL_CORRECTION_MAX = 0.006
FILTER_CUTOFF_HZ = 20.0

KP_SE3 = np.array([225.0, 225.0, 225.0, 144.0, 144.0, 144.0])
KD_SE3 = 2.0 * np.sqrt(KP_SE3)
JOINT_VELOCITY_LIMIT = 2.0
JOINT_ACCELERATION_LIMIT = 30.0

EXPECTED_CONTACT_PAIR = frozenset(
    ("inspection_probe_tip_collision", "inspection_workpiece")
)


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise RuntimeError("零向量无法归一化")
    return vector / norm


def quintic(u):
    u = float(np.clip(u, 0.0, 1.0))
    position = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    acceleration = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    return position, velocity, acceleration


def object_id(model, object_type, name):
    index = mujoco.mj_name2id(model, object_type, name)
    if index < 0:
        raise RuntimeError(f"MuJoCo找不到对象: {name}")
    return index


def solve_pose_ik(model, frame_id, q_seed, position, rotation):
    data = model.createData()
    q = q_seed.copy()
    for _ in range(250):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pose = data.oMf[frame_id]
        error = np.concatenate(
            (position - pose.translation, pin.log3(rotation @ pose.rotation.T))
        )
        if np.linalg.norm(error[:3]) < 2e-7 and np.linalg.norm(error[3:]) < 2e-7:
            return q
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        jacobian = pin.getFrameJacobian(
            model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        damping = 1e-4 if singular_values[-1] > 0.05 else 0.02
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error
        )
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0.10:
            delta *= 0.10 / delta_norm
        q = pin.integrate(model, q, delta)
    raise RuntimeError("预接触位姿IK未收敛")


def is_descendant(model, body_id, root_body_id):
    current = int(body_id)
    while current > 0:
        if current == root_body_id:
            return True
        current = int(model.body_parentid[current])
    return False


def unexpected_contacts(model, data):
    bad = []
    for index in range(data.ncon):
        contact = data.contact[index]
        name1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
        )
        name2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
        )
        pair = frozenset((name1, name2))
        if pair != EXPECTED_CONTACT_PAIR:
            bad.append(tuple(sorted((str(name1), str(name2)))))
    return bad


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="打开MuJoCo Viewer")
    return parser.parse_args()


def main():
    args = parse_args()
    geometry = np.load(GEOMETRY_PATH)
    ik_data = np.load(IK_PATH)
    system = load_robot_system(MJCF_PATH, URDF_PATH)
    mj_model, mj_data = system.mj_model, system.mj_data
    model, robot = system.pin_model, system.robot

    p_surface = geometry["position"][0].copy()
    n_out = normalize(geometry["surface_normal"][0])
    rotation_target = geometry["rotation"][0].copy()
    if np.dot(rotation_target[:, 2], -n_out) < 1.0 - 1e-10:
        raise RuntimeError("工具+Z轴与曲面内法向不一致")

    frame_id = model.getFrameId(FRAME_NAME)
    sensor_frame_id = model.getFrameId(SENSOR_FRAME_NAME)
    if frame_id >= model.nframes or sensor_frame_id >= model.nframes:
        raise RuntimeError("Pinocchio缺少TCP或FT300 frame")

    ft_site_id = object_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "ft300_site")
    tip_site_id = object_id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip_site"
    )
    force_sensor_id = object_id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft300_force"
    )
    torque_sensor_id = object_id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ft300_torque"
    )
    force_adr = int(mj_model.sensor_adr[force_sensor_id])
    torque_adr = int(mj_model.sensor_adr[torque_sensor_id])

    sensor_body_id = int(mj_model.site_bodyid[ft_site_id])
    payload_body_ids = [
        body_id
        for body_id in range(mj_model.nbody)
        if is_descendant(mj_model, body_id, sensor_body_id)
    ]
    payload_mass = float(sum(mj_model.body_mass[i] for i in payload_body_ids))

    q_surface = ik_data["q"][0].copy()
    p_precontact = p_surface + PRECONTACT_GAP * n_out
    q_precontact = solve_pose_ik(
        model, frame_id, q_surface, p_precontact, rotation_target
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
        payload_com = (
            weighted_com / payload_mass
            if payload_mass > 1e-12
            else mj_data.site_xpos[ft_site_id]
        )
        sensor_position = mj_data.site_xpos[ft_site_id]
        force_world = -payload_mass * np.asarray(mj_model.opt.gravity)
        torque_world = np.cross(payload_com - sensor_position, force_world)
        rotation_sensor = mj_data.site_xmat[ft_site_id].reshape(3, 3)
        return np.concatenate(
            (rotation_sensor.T @ force_world, rotation_sensor.T @ torque_world)
        )

    def wrench_to_external_world(wrench_sensor):
        rotation_sensor = mj_data.site_xmat[ft_site_id].reshape(3, 3)
        return np.concatenate(
            (
                -rotation_sensor @ wrench_sensor[:3],
                -rotation_sensor @ wrench_sensor[3:],
            )
        )

    formulation = tsid.InverseDynamicsFormulationAccForce(
        "ur5-ft300-contact", robot, False
    )
    formulation.computeProblemData(0.0, q_precontact, np.zeros(model.nv))

    se3_task = tsid.TaskSE3Equality("tcp-pose", robot, FRAME_NAME)
    se3_task.useLocalFrame(False)
    se3_task.setMask(np.ones(6))
    se3_task.setKp(KP_SE3)
    se3_task.setKd(KD_SE3)
    se3_task.setReference(se3_sample(pin.SE3(rotation_target, p_precontact)))

    dt = float(mj_model.opt.timestep)
    joint_bounds = tsid.TaskJointPosVelAccBounds(
        "joint-bounds", robot, dt, False
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
    actuation_bounds = tsid.TaskActuationBounds("motor-bounds", robot)
    actuation_bounds.setMask(np.ones(robot.na))
    actuation_bounds.setBounds(motor_lower, motor_upper)

    formulation.addMotionTask(joint_bounds, 1.0, 0, 0.0)
    formulation.addActuationTask(actuation_bounds, 1.0, 0, 0.0)
    formulation.addMotionTask(se3_task, 1.0, 1, 0.0)

    solver = tsid.SolverHQuadProgFast("ur5-contact-qpoases")
    solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

    filter_tau = 1.0 / (2.0 * np.pi * FILTER_CUTOFF_HZ)
    filter_alpha = dt / (filter_tau + dt)
    filtered_wrench = np.zeros(6)
    bias_sum = np.zeros(6)
    bias_count = 0
    bias_wrench = None
    force_sign = None
    contact_time = None
    withdraw_time = None
    withdraw_start_offset = None
    normal_correction = 0.0
    normal_velocity = 0.0
    normal_acceleration = 0.0

    force_errors = []
    tcp_errors = []
    normal_errors = []
    history_time = []
    history_phase = []
    history_force_desired = []
    history_force_measured = []
    history_normal_correction = []
    max_force = 0.0
    max_ctrl = 0.0
    max_bound_violation = 0.0
    max_dynamics_residual = 0.0
    qp_failures = 0
    unexpected_contact_count = 0
    solve_times = []
    completed = False
    dynamics_data = model.createData()
    jacobian_data = model.createData()

    print("========== 104 TSID/HQP + FT300 5 N Contact ==========")
    print(f"Pre-contact gap 预接触间隙: {PRECONTACT_GAP * 1000:.1f} mm")
    print(f"Payload mass 传感器下游质量: {payload_mass:.4f} kg")
    print(f"Force target 目标力: {DESIRED_FORCE:.1f} N")
    print("Contact model 接触模型: MuJoCo only（TSID不添加刚性接触）")
    print("Post-solve clipping 求解后裁剪: DISABLED")

    if args.viewer:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(mj_model, mj_data)
    else:
        context = nullcontext(None)

    max_time = (
        BIAS_END
        + APPROACH_DURATION
        + FORCE_RAMP_DURATION
        + FORCE_HOLD_DURATION
        + WITHDRAW_DURATION
        + 1.0
    )
    last_print = -1.0

    with context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = np.array([0.02, -0.18, 0.82])
            viewer.cam.distance = 1.85
            viewer.cam.azimuth = 142.0
            viewer.cam.elevation = -24.0

        while mj_data.time < max_time:
            if viewer is not None and not viewer.is_running():
                break
            step_wall_start = time.perf_counter()
            t = float(mj_data.time)

            raw_wrench = read_raw_wrench()
            gravity_wrench = payload_gravity_wrench()
            if BIAS_START <= t < BIAS_END:
                bias_sum += raw_wrench - gravity_wrench
                bias_count += 1
            if bias_wrench is None and t >= BIAS_END:
                if bias_count == 0:
                    raise RuntimeError("没有采集到FT300零偏样本")
                bias_wrench = bias_sum / bias_count
                print("Bias calibration FT300零偏标定: completed 完成")

            compensated = (
                np.zeros(6)
                if bias_wrench is None
                else raw_wrench - gravity_wrench - bias_wrench
            )
            filtered_wrench += filter_alpha * (compensated - filtered_wrench)
            external_wrench = wrench_to_external_world(filtered_wrench)
            projected_force = float(np.dot(external_wrench[:3], n_out))
            measured_force = (
                abs(projected_force)
                if force_sign is None
                else max(0.0, force_sign * projected_force)
            )
            max_force = max(max_force, measured_force)

            desired_force = 0.0
            desired_velocity = np.zeros(3)
            desired_acceleration = np.zeros(3)

            if t < BIAS_END:
                phase = 0
                phase_text = "bias/hold 零偏标定"
                desired_position = p_precontact.copy()
            elif contact_time is None:
                phase = 1
                phase_text = "approach 接近"
                u = (t - BIAS_END) / APPROACH_DURATION
                s, sd, sdd = quintic(u)
                delta = -MAX_APPROACH_PENETRATION - PRECONTACT_GAP
                offset = PRECONTACT_GAP + s * delta
                offset_velocity = sd * delta / APPROACH_DURATION
                offset_acceleration = sdd * delta / APPROACH_DURATION**2
                desired_position = p_surface + offset * n_out
                desired_velocity = offset_velocity * n_out
                desired_acceleration = offset_acceleration * n_out

                if bias_wrench is not None and measured_force >= CONTACT_THRESHOLD:
                    contact_time = t
                    force_sign = 1.0 if projected_force >= 0.0 else -1.0
                    actual_tip = mj_data.site_xpos[tip_site_id].copy()
                    normal_correction = float(
                        np.clip(
                            -np.dot(actual_tip - p_surface, n_out),
                            0.0,
                            NORMAL_CORRECTION_MAX,
                        )
                    )
                    normal_velocity = 0.0
                    print(
                        f"Contact detected 检测到接触: t={t:.3f}s, "
                        f"F={measured_force:.3f}N, sign={force_sign:+.0f}"
                    )
            elif withdraw_time is None:
                elapsed = t - contact_time
                if elapsed < FORCE_RAMP_DURATION:
                    phase = 2
                    phase_text = "force ramp 力建立"
                    desired_force = DESIRED_FORCE * quintic(
                        elapsed / FORCE_RAMP_DURATION
                    )[0]
                elif elapsed < FORCE_RAMP_DURATION + FORCE_HOLD_DURATION:
                    phase = 3
                    phase_text = "force hold 恒力保持"
                    desired_force = DESIRED_FORCE
                else:
                    withdraw_time = t
                    withdraw_start_offset = -normal_correction
                    phase = 4
                    phase_text = "withdraw 撤离"

                if withdraw_time is None:
                    force_error = desired_force - measured_force
                    normal_acceleration = np.clip(
                        (force_error - FORCE_DAMPING * normal_velocity)
                        / FORCE_MASS,
                        -NORMAL_ACCEL_LIMIT,
                        NORMAL_ACCEL_LIMIT,
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
                            0.0,
                            NORMAL_CORRECTION_MAX,
                        )
                    )
                    if normal_correction in (0.0, NORMAL_CORRECTION_MAX):
                        normal_velocity = 0.0
                    desired_position = p_surface - normal_correction * n_out
                    desired_velocity = -normal_velocity * n_out
                    desired_acceleration = -normal_acceleration * n_out
                else:
                    desired_position = p_surface + withdraw_start_offset * n_out
            else:
                phase = 4
                phase_text = "withdraw 撤离"
                u = (t - withdraw_time) / WITHDRAW_DURATION
                s, sd, sdd = quintic(u)
                delta = WITHDRAW_GAP - withdraw_start_offset
                offset = withdraw_start_offset + s * delta
                desired_position = p_surface + offset * n_out
                desired_velocity = sd * delta / WITHDRAW_DURATION * n_out
                desired_acceleration = sdd * delta / WITHDRAW_DURATION**2 * n_out
                if u >= 1.0:
                    completed = True

            reference = se3_sample(pin.SE3(rotation_target, desired_position))
            reference.derivative(np.concatenate((desired_velocity, np.zeros(3))))
            reference.second_derivative(
                np.concatenate((desired_acceleration, np.zeros(3)))
            )
            se3_task.setReference(reference)

            q, v = read_state(system)
            pin.computeJointJacobians(model, jacobian_data, q)
            pin.updateFramePlacements(model, jacobian_data)
            sensor_jacobian = pin.getFrameJacobian(
                model,
                jacobian_data,
                sensor_frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            external_generalized_force = sensor_jacobian.T @ external_wrench

            passive = np.zeros(robot.na)
            for item in system.joints:
                passive[item.pin_v] = mj_data.qfrc_passive[item.mj_v]
            # tau_tsid - J^T Fext - passive is the actual motor command.
            actuation_bounds.setBounds(
                motor_lower + external_generalized_force + passive,
                motor_upper + external_generalized_force + passive,
            )

            problem = formulation.computeProblemData(t, q, v)
            solve_start = time.perf_counter()
            solution = solver.solve(problem)
            solve_times.append((time.perf_counter() - solve_start) * 1e6)
            if solution.status != 0:
                qp_failures += 1
                print(f"QP failed QP失败: t={t:.3f}s status={solution.status}")
                break

            tau_tsid = np.asarray(
                formulation.getActuatorForces(solution)
            ).reshape(-1)
            ddq = np.asarray(formulation.getAccelerations(solution)).reshape(-1)
            max_dynamics_residual = max(
                max_dynamics_residual,
                inverse_dynamics_residual(
                    system, dynamics_data, q, v, ddq, tau_tsid
                ),
            )
            motor_generalized_force = tau_tsid - external_generalized_force
            ctrl, bound_violation = tau_to_ctrl(
                system, motor_generalized_force, clip=False
            )
            max_bound_violation = max(max_bound_violation, bound_violation)
            max_ctrl = max(max_ctrl, float(np.max(np.abs(ctrl))))
            mj_data.ctrl[:] = ctrl
            mujoco.mj_step(mj_model, mj_data)

            bad_contacts = unexpected_contacts(mj_model, mj_data)
            if bad_contacts:
                unexpected_contact_count += 1
                raise RuntimeError(f"出现非预期碰撞: {bad_contacts}")

            actual_position = mj_data.site_xpos[tip_site_id].copy()
            tcp_error = float(np.linalg.norm(desired_position - actual_position))
            actual_rotation = mj_data.site_xmat[tip_site_id].reshape(3, 3)
            normal_error = float(
                np.arccos(
                    np.clip(np.dot(actual_rotation[:, 2], -n_out), -1.0, 1.0)
                )
            )

            if (
                contact_time is not None
                and withdraw_time is None
                and t >= contact_time + FORCE_RAMP_DURATION + 1.0
            ):
                force_errors.append(abs(DESIRED_FORCE - measured_force))
                tcp_errors.append(tcp_error)
                normal_errors.append(normal_error)

            history_time.append(t)
            history_phase.append(phase)
            history_force_desired.append(desired_force)
            history_force_measured.append(measured_force)
            history_normal_correction.append(normal_correction)

            if t - last_print >= 1.0:
                last_print = t
                print(
                    f"t={t:5.2f}s | {phase_text:20s} | "
                    f"F={desired_force:4.2f}/{measured_force:4.2f}N | "
                    f"dn={normal_correction * 1000:+.3f}mm | "
                    f"TCP={tcp_error * 1000:.3f}mm | "
                    f"tau={np.max(np.abs(ctrl)):.2f}Nm"
                )

            if viewer is not None:
                viewer.sync()
                remaining = dt - (time.perf_counter() - step_wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            if completed:
                break

    solve_times = np.asarray(solve_times)
    force_mean_error = float(np.mean(force_errors)) if force_errors else np.inf
    force_max_error = float(np.max(force_errors)) if force_errors else np.inf
    tcp_mean_error = float(np.mean(tcp_errors)) if tcp_errors else np.inf
    normal_max_error = float(np.max(normal_errors)) if normal_errors else np.inf
    final_contact_count = int(mj_data.ncon)

    passed = (
        completed
        and contact_time is not None
        and qp_failures == 0
        and unexpected_contact_count == 0
        and max_bound_violation < 1e-7
        and force_mean_error < 0.6
        and force_max_error < 3.0
        and tcp_mean_error < 8e-4
        and normal_max_error < np.deg2rad(0.5)
        and final_contact_count == 0
        and max_dynamics_residual < 1e-8
    )

    np.savez_compressed(
        OUTPUT_PATH,
        time=np.asarray(history_time),
        phase=np.asarray(history_phase, dtype=int),
        desired_force=np.asarray(history_force_desired),
        measured_force=np.asarray(history_force_measured),
        normal_correction=np.asarray(history_normal_correction),
        contact_time=np.nan if contact_time is None else contact_time,
        passed=passed,
    )

    print("\n========== 104 Result 结果 ==========")
    print(f"Contact/withdraw completed 接触/撤离完成: {contact_time is not None}/{completed}")
    print(f"Force mean/max error 恒力平均/最大误差: {force_mean_error:.3f}/{force_max_error:.3f} N")
    print(f"TCP mean error TCP平均误差: {tcp_mean_error * 1000:.3f} mm")
    print(f"Tool-normal max error 工具法向最大误差: {np.rad2deg(normal_max_error):.3f} deg")
    print(f"Maximum measured force 最大实测力: {max_force:.3f} N")
    print(f"Maximum motor torque 最大电机力矩: {max_ctrl:.3f} Nm")
    print(f"Motor-bound violation 电机约束违规: {max_bound_violation:.3e} Nm")
    print(f"QP failures / unexpected contacts: {qp_failures}/{unexpected_contact_count}")
    print(f"Final contacts 撤离后接触数: {final_contact_count}")
    print(f"Dynamics residual 动力学残差: {max_dynamics_residual:.3e}")
    if solve_times.size:
        print(
            f"QP solve mean/P99: {np.mean(solve_times):.1f}/"
            f"{np.percentile(solve_times, 99):.1f} us"
        )
    print(f"Saved 保存: {OUTPUT_PATH.name}")
    print(f"Result 结果: {'PASS 通过' if passed else 'FAIL 失败'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
