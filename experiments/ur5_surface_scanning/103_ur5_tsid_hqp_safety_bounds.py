#!/usr/bin/env python3
"""103 - Enforce joint and actuator safety bounds inside the TSID HQP.

An intentionally aggressive posture command is limited by hard position,
velocity, acceleration and conservative software-torque bounds.  No torque
clipping is performed after the solve: any violation therefore fails the test.
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
    configuration_from_degrees,
    inverse_dynamics_residual,
    load_robot_system,
    posture_sample,
    read_state,
    reset_mujoco,
    tau_to_ctrl,
)


HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "ur5_ft300_inspection.xml"
URDF_PATH = HERE / "ur5_ft300_inspection.urdf"

INITIAL_DEG = np.array([0.0, -60.0, 90.0, -30.0, 45.0, 0.0])
TARGET_DEG = np.array([55.0, -115.0, 135.0, -85.0, -40.0, 80.0])
VELOCITY_LIMIT = 0.25       # rad/s
ACCELERATION_LIMIT = 2.0    # rad/s^2
# Conservative software limits, stricter than the MJCF physical limits.
SOFTWARE_TORQUE_LIMIT = np.array([20.0, 36.0, 16.0, 5.0, 5.0, 3.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="打开MuJoCo Viewer")
    parser.add_argument("--duration", type=float, default=6.0, help="仿真时长/s")
    return parser.parse_args()


def passive_generalized_force(system):
    value = np.zeros(system.robot.na)
    for item in system.joints:
        value[item.pin_v] = system.mj_data.qfrc_passive[item.mj_v]
    return value


def main():
    args = parse_args()
    system = load_robot_system(MJCF_PATH, URDF_PATH)
    model = system.pin_model
    robot = system.robot
    q_initial = configuration_from_degrees(system, INITIAL_DEG)
    q_target = configuration_from_degrees(system, TARGET_DEG)
    reset_mujoco(system, q_initial)

    formulation = tsid.InverseDynamicsFormulationAccForce(
        "ur5-safety-hqp", robot, False
    )
    formulation.computeProblemData(0.0, q_initial, np.zeros(model.nv))

    posture = tsid.TaskJointPosture("aggressive-posture", robot)
    posture.setMask(np.ones(robot.na))
    posture.setKp(100.0 * np.ones(robot.na))
    posture.setKd(20.0 * np.ones(robot.na))
    posture.setReference(posture_sample(model, q_target))

    dt = float(system.mj_model.opt.timestep)
    joint_bounds = tsid.TaskJointPosVelAccBounds(
        "joint-pos-vel-acc-bounds", robot, dt, False
    )
    joint_bounds.setMask(np.ones(robot.na))
    joint_bounds.setPositionBounds(
        model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    )
    joint_bounds.setVelocityBounds(VELOCITY_LIMIT * np.ones(robot.na))
    joint_bounds.setAccelerationBounds(ACCELERATION_LIMIT * np.ones(robot.na))
    joint_bounds.setImposeBounds(True, True, True, True)

    physical_lower, physical_upper = actuator_bounds(system)
    safe_lower = np.maximum(physical_lower, -SOFTWARE_TORQUE_LIMIT)
    safe_upper = np.minimum(physical_upper, SOFTWARE_TORQUE_LIMIT)
    actuation_bounds = tsid.TaskActuationBounds("actuation-bounds", robot)
    actuation_bounds.setMask(np.ones(robot.na))
    actuation_bounds.setBounds(safe_lower, safe_upper)

    formulation.addMotionTask(joint_bounds, 1.0, 0, 0.0)
    formulation.addActuationTask(actuation_bounds, 1.0, 0, 0.0)
    formulation.addMotionTask(posture, 1.0, 1, 0.0)

    solver = tsid.SolverHQuadProgFast("ur5-safety-qpoases")
    solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

    print("========== 103 UR5 TSID/HQP Safety Bounds ==========")
    print(f"Velocity limit 速度限制: ±{VELOCITY_LIMIT:.3f} rad/s")
    print(f"Acceleration limit 加速度限制: ±{ACCELERATION_LIMIT:.3f} rad/s²")
    print(f"Software torque limits 软件力矩限制: {SOFTWARE_TORQUE_LIMIT.tolist()} Nm")
    print("Post-solve clipping 求解后裁剪: DISABLED")
    print(
        f"HQP: nVar={formulation.nVar}, nEq={formulation.nEq}, nIn={formulation.nIn}"
    )

    steps = int(np.ceil(args.duration / dt))
    dynamics_data = model.createData()
    previous_v = np.zeros(model.nv)
    max_speed = np.zeros(model.nv)
    max_acceleration = np.zeros(model.nv)
    max_abs_ctrl = np.zeros(model.nv)
    min_position_margin = np.inf
    max_physical_violation = 0.0
    max_software_violation = 0.0
    max_dynamics_residual = 0.0
    active_velocity_steps = 0
    active_torque_steps = 0
    qp_failures = 0
    solve_times = []

    if args.viewer:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(system.mj_model, system.mj_data)
    else:
        context = nullcontext(None)

    with context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = np.array([-0.15, 0.05, 0.75])
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -20.0

        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            wall_start = time.perf_counter()
            q, v = read_state(system)

            # TSID constrains complete generalized actuation tau.  MuJoCo's
            # motor command is tau - qfrc_passive, so shift the bounds every
            # cycle to make the HQP constrain the actual data.ctrl value.
            passive = passive_generalized_force(system)
            actuation_bounds.setBounds(
                safe_lower + passive, safe_upper + passive
            )

            problem = formulation.computeProblemData(system.mj_data.time, q, v)
            solve_start = time.perf_counter()
            solution = solver.solve(problem)
            solve_times.append((time.perf_counter() - solve_start) * 1e6)
            if solution.status != 0:
                qp_failures += 1
                system.mj_data.ctrl[:] = 0.0
                print(f"QP failed: step={step}, status={solution.status}")
                break

            tau = np.asarray(formulation.getActuatorForces(solution)).reshape(-1)
            ddq = np.asarray(formulation.getAccelerations(solution)).reshape(-1)
            max_dynamics_residual = max(
                max_dynamics_residual,
                inverse_dynamics_residual(system, dynamics_data, q, v, ddq, tau),
            )

            # Deliberately no clipping: constraints must already be satisfied.
            ctrl, physical_violation = tau_to_ctrl(system, tau, clip=False)
            software_violation = float(
                np.max(np.maximum(np.abs(ctrl) - SOFTWARE_TORQUE_LIMIT, 0.0))
            )
            max_physical_violation = max(max_physical_violation, physical_violation)
            max_software_violation = max(max_software_violation, software_violation)
            max_abs_ctrl = np.maximum(max_abs_ctrl, np.abs(ctrl))
            active_torque_steps += int(
                np.any(np.abs(ctrl) >= SOFTWARE_TORQUE_LIMIT - 0.02)
            )

            system.mj_data.ctrl[:] = ctrl
            mujoco.mj_step(system.mj_model, system.mj_data)

            q_after, v_after = read_state(system)
            measured_acceleration = (v_after - previous_v) / dt
            previous_v = v_after.copy()
            max_speed = np.maximum(max_speed, np.abs(v_after))
            max_acceleration = np.maximum(
                max_acceleration, np.abs(measured_acceleration)
            )
            active_velocity_steps += int(
                np.any(np.abs(v_after) >= VELOCITY_LIMIT - 5e-4)
            )
            position_margin = np.minimum(
                q_after - model.lowerPositionLimit,
                model.upperPositionLimit - q_after,
            )
            min_position_margin = min(
                min_position_margin, float(np.min(position_margin))
            )

            if viewer is not None:
                viewer.sync()
                remaining = dt - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)

            if step % max(1, int(round(1.0 / dt))) == 0:
                error_deg = np.rad2deg(
                    np.linalg.norm(pin.difference(model, q_after, q_target))
                )
                print(
                    f"t={system.mj_data.time:5.2f}s | error={error_deg:7.2f} deg | "
                    f"vmax={np.max(np.abs(v_after)):.4f} | "
                    f"taumax={np.max(np.abs(ctrl)):.3f}"
                )

    solve_times = np.asarray(solve_times)
    # The first finite difference contains the initial acceleration and is
    # valid, but integration/discretization allows a small numerical margin.
    passed = (
        qp_failures == 0
        and max_physical_violation < 1e-8
        and max_software_violation < 1e-6
        and np.max(max_speed) <= VELOCITY_LIMIT + 2e-3
        and np.max(max_acceleration) <= ACCELERATION_LIMIT + 1e-2
        and min_position_margin >= -1e-8
        and active_velocity_steps > 0
        and active_torque_steps > 0
        and max_dynamics_residual < 1e-8
    )

    print("\n========== 103 Result 结果 ==========")
    print(f"Max joint speeds 最大关节速度: {max_speed.round(6).tolist()} rad/s")
    print(
        f"Max measured acceleration 最大实测加速度: "
        f"{max_acceleration.round(4).tolist()} rad/s²"
    )
    print(f"Max motor commands 最大电机指令: {max_abs_ctrl.round(4).tolist()} Nm")
    print(f"Minimum position margin 最小位置裕量: {min_position_margin:.6f} rad")
    print(f"Physical/software torque violation 力矩违规: {max_physical_violation:.3e}/{max_software_violation:.3e} Nm")
    print(f"Active velocity/torque steps 激活约束步数: {active_velocity_steps}/{active_torque_steps}")
    print(f"QP failures QP失败: {qp_failures}")
    print(f"Dynamics residual 动力学残差: {max_dynamics_residual:.3e}")
    if solve_times.size:
        print(
            f"QP solve mean/P99: {np.mean(solve_times):.1f}/"
            f"{np.percentile(solve_times, 99):.1f} us"
        )
    print(f"Result 结果: {'PASS 通过' if passed else 'FAIL 失败'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
