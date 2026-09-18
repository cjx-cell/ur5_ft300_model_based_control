#!/usr/bin/env python3
"""102 - Weighted 5D Cartesian + posture TSID control in MuJoCo.

The primary task controls TCP XYZ and world-aligned roll/pitch while leaving
one rotational degree of freedom available.  A low-weight posture task uses
the remaining freedom.  This is the task structure needed later for
surface scanning: tangential motion and tool tilt can outrank posture/yaw.
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
    configuration_from_degrees,
    frame_pose,
    inverse_dynamics_residual,
    load_robot_system,
    posture_sample,
    read_state,
    reset_mujoco,
    se3_sample,
    tau_to_ctrl,
)


HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "ur5_ft300_inspection.xml"
URDF_PATH = HERE / "ur5_ft300_inspection.urdf"
FRAME_NAME = "inspection_tip"

INITIAL_DEG = np.array([0.0, -60.0, 90.0, -30.0, 45.0, 0.0])
POSTURE_CHANGE_DEG = np.array([8.0, -6.0, 7.0, 5.0, -5.0, 20.0])
POSITION_OFFSET = np.array([0.020, -0.015, 0.015])
WORLD_TILT_DEG = np.array([4.0, -5.0, 0.0])
SE3_MASK = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="打开MuJoCo Viewer")
    parser.add_argument("--duration", type=float, default=6.0, help="仿真时长/s")
    return parser.parse_args()


def main():
    args = parse_args()
    system = load_robot_system(MJCF_PATH, URDF_PATH)
    model = system.pin_model
    robot = system.robot

    frame_id = model.getFrameId(FRAME_NAME)
    if frame_id >= model.nframes:
        raise RuntimeError(f"Pinocchio找不到frame: {FRAME_NAME}")

    q_initial = configuration_from_degrees(system, INITIAL_DEG)
    reset_mujoco(system, q_initial)

    kinematics_data = model.createData()
    initial_pose = frame_pose(system, kinematics_data, frame_id, q_initial)
    target_pose = pin.SE3(
        pin.exp3(np.deg2rad(WORLD_TILT_DEG)) @ initial_pose.rotation,
        initial_pose.translation + POSITION_OFFSET,
    )
    q_posture = pin.integrate(
        model, q_initial, np.deg2rad(POSTURE_CHANGE_DEG)
    )

    formulation = tsid.InverseDynamicsFormulationAccForce(
        "ur5-se3-hierarchy", robot, False
    )
    formulation.computeProblemData(0.0, q_initial, np.zeros(model.nv))

    se3_task = tsid.TaskSE3Equality("tcp-5d", robot, FRAME_NAME)
    se3_task.useLocalFrame(False)
    se3_task.setMask(SE3_MASK)
    se3_kp = np.array([100.0, 100.0, 100.0, 64.0, 64.0, 0.0])
    se3_task.setKp(se3_kp)
    se3_task.setKd(2.0 * np.sqrt(se3_kp))
    se3_task.setReference(se3_sample(target_pose))

    posture_task = tsid.TaskJointPosture("posture-secondary", robot)
    posture_task.setMask(np.ones(robot.na))
    posture_task.setKp(16.0 * np.ones(robot.na))
    posture_task.setKd(8.0 * np.ones(robot.na))
    posture_task.setReference(posture_sample(model, q_posture))

    # TSID 1.10's Python formulation supports hard constraints at level 0 and
    # weighted motion costs at level 1.  Passing level 2 to this binding causes
    # a native crash, so strict ordering inside the soft level is expressed by
    # a deliberately separated weight ratio.
    formulation.addMotionTask(se3_task, 1.0, 1, 0.0)
    formulation.addMotionTask(posture_task, 1.0e-4, 1, 0.0)

    solver = tsid.SolverHQuadProgFast("ur5-hqp")
    solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

    dt = float(system.mj_model.opt.timestep)
    steps = int(np.ceil(args.duration / dt))
    dynamics_data = model.createData()
    solve_times = []
    qp_failures = 0
    saturation_steps = 0
    max_dynamics_residual = 0.0
    max_ctrl = 0.0

    print("========== 102 UR5 TSID Cartesian Multi-task ==========")
    print("Primary 主任务: TCP XYZ + world roll/pitch (5D), weight 1")
    print("Secondary 次任务: joint posture, weight 1e-4")
    print(f"Position target 位置变化: {(POSITION_OFFSET * 1000).tolist()} mm")
    print(f"Tilt target 姿态变化: {WORLD_TILT_DEG.tolist()} deg")
    print(
        f"HQP: nVar={formulation.nVar}, nEq={formulation.nEq}, "
        f"nIn={formulation.nIn}"
    )

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
            ctrl, violation = tau_to_ctrl(system, tau, clip=True)
            saturation_steps += int(violation > 1e-10)
            max_ctrl = max(max_ctrl, float(np.max(np.abs(ctrl))))
            system.mj_data.ctrl[:] = ctrl
            mujoco.mj_step(system.mj_model, system.mj_data)

            if viewer is not None:
                viewer.sync()
                remaining = dt - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)

            if step % max(1, int(round(1.0 / dt))) == 0:
                q_now, _ = read_state(system)
                pose = frame_pose(system, kinematics_data, frame_id, q_now)
                pos_mm = np.linalg.norm(target_pose.translation - pose.translation) * 1000
                rot_world = pin.log3(target_pose.rotation @ pose.rotation.T)
                tilt_deg = np.rad2deg(np.linalg.norm(rot_world[:2]))
                print(
                    f"t={system.mj_data.time:5.2f}s | pos={pos_mm:7.3f} mm | "
                    f"tilt={tilt_deg:7.3f} deg | ctrl={np.max(np.abs(ctrl)):7.3f} Nm"
                )

    q_final, v_final = read_state(system)
    final_pose = frame_pose(system, kinematics_data, frame_id, q_final)
    position_error = float(
        np.linalg.norm(target_pose.translation - final_pose.translation)
    )
    rotation_error_world = pin.log3(
        target_pose.rotation @ final_pose.rotation.T
    )
    tilt_error = float(np.linalg.norm(rotation_error_world[:2]))
    free_yaw_error = float(abs(rotation_error_world[2]))
    posture_error = float(np.linalg.norm(pin.difference(model, q_final, q_posture)))
    final_speed = float(np.linalg.norm(v_final))
    solve_times = np.asarray(solve_times)

    passed = (
        qp_failures == 0
        and saturation_steps == 0
        and position_error < 5e-5
        and tilt_error < np.deg2rad(0.02)
        and final_speed < 0.01
        and max_dynamics_residual < 1e-8
    )

    print("\n========== 102 Result 结果 ==========")
    print(f"TCP position error TCP位置误差: {position_error * 1000:.6f} mm")
    print(f"Controlled tilt error 受控倾斜误差: {np.rad2deg(tilt_error):.6f} deg")
    print(f"Free yaw difference 自由偏航差: {np.rad2deg(free_yaw_error):.6f} deg")
    print(f"Secondary posture error 次任务姿态误差: {np.rad2deg(posture_error):.4f} deg")
    print(f"Final speed 最终速度: {final_speed:.3e} rad/s")
    print(f"Max ctrl 最大指令: {max_ctrl:.3f} Nm")
    print(f"Saturation/QP failures 饱和/QP失败: {saturation_steps}/{qp_failures}")
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
