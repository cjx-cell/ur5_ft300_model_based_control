#!/usr/bin/env python3
"""101 - Close a TSID posture-control torque loop around MuJoCo.

Unlike the archived 80-series examples, TSID does not integrate its own ideal
state here.  MuJoCo is the plant: each cycle reads q/dq from MuJoCo, solves the
TSID HQP, compensates MuJoCo passive forces, and sends torque to ``data.ctrl``.

Run headless (default):
    python 101_ur5_tsid_mujoco_posture.py

Run with visualization:
    python 101_ur5_tsid_mujoco_posture.py --viewer
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*to-Python converter.*already registered.*",
)

import mujoco
import numpy as np
import pinocchio as pin
import tsid

HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "ur5_ft300_inspection.xml"
URDF_PATH = HERE / "ur5_ft300_inspection.urdf"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

INITIAL_DEG = np.array([0.0, -60.0, 90.0, -30.0, 45.0, 0.0])
TARGET_DEG = np.array([12.0, -70.0, 100.0, -20.0, 35.0, 15.0])

DURATION = 5.0
KP = np.array([36.0, 36.0, 36.0, 25.0, 25.0, 16.0])
KD = 2.0 * np.sqrt(KP)


def build_joint_mapping(mj_model, pin_model):
    mapping = []
    for name in JOINT_NAMES:
        mj_joint = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        pin_joint = pin_model.getJointId(name)
        if mj_joint < 0 or pin_joint == 0:
            raise RuntimeError(f"关节映射失败: {name}")
        joint = pin_model.joints[pin_joint]
        mapping.append(
            {
                "name": name,
                "mj_joint": mj_joint,
                "mj_q": int(mj_model.jnt_qposadr[mj_joint]),
                "mj_v": int(mj_model.jnt_dofadr[mj_joint]),
                "pin_q": int(joint.idx_q),
                "pin_v": int(joint.idx_v),
            }
        )
    return mapping


def mj_state_to_pin(mj_data, pin_model, mapping):
    q = pin.neutral(pin_model)
    v = np.zeros(pin_model.nv)
    for item in mapping:
        q[item["pin_q"]] = mj_data.qpos[item["mj_q"]]
        v[item["pin_v"]] = mj_data.qvel[item["mj_v"]]
    return q, v


def pin_q_to_mj(q, mj_model, mapping):
    q_mj = np.zeros(mj_model.nq)
    for item in mapping:
        q_mj[item["mj_q"]] = q[item["pin_q"]]
    return q_mj


def generalized_tau_to_ctrl(tau, mj_data, mj_model, mapping):
    """Map desired generalized force to direct-drive MuJoCo controls.

    MuJoCo adds qfrc_passive separately to its equation of motion, whereas
    TSID returns the complete generalized actuation.  It must therefore be
    removed before sending ctrl.  This model has one direct motor per joint.
    """
    ctrl = np.zeros(mj_model.nu)
    unclipped = np.zeros(mj_model.nu)

    for actuator_id, item in enumerate(mapping):
        transmission_joint = int(mj_model.actuator_trnid[actuator_id, 0])
        gear = float(mj_model.actuator_gear[actuator_id, 0])
        if transmission_joint != item["mj_joint"] or abs(gear) < 1e-12:
            raise RuntimeError(
                f"执行器{actuator_id}不是预期的直接关节传动: {item['name']}"
            )

        requested = (
            tau[item["pin_v"]] - mj_data.qfrc_passive[item["mj_v"]]
        ) / gear
        unclipped[actuator_id] = requested

        if mj_model.actuator_ctrllimited[actuator_id]:
            low, high = mj_model.actuator_ctrlrange[actuator_id]
            requested = np.clip(requested, low, high)
        ctrl[actuator_id] = requested

    return ctrl, unclipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="打开MuJoCo可视化；默认进行无界面自动验证",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION,
        help=f"仿真时长，默认{DURATION:g}秒",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (MJCF_PATH, URDF_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)

    package_dirs = pin.StdVec_StdString()
    package_dirs.append(str(URDF_PATH.parent))
    robot = tsid.RobotWrapper(str(URDF_PATH), package_dirs, False)
    model = robot.model()
    mapping = build_joint_mapping(mj_model, model)

    if not (
        mj_model.nq == mj_model.nv == mj_model.nu == 6
        and model.nq == model.nv == int(robot.na) == 6
    ):
        raise RuntimeError("本实验要求固定基、完全驱动的6自由度UR5")

    # TSID adds reflected motor inertia through RobotWrapper's rotor API.  A
    # direct write to model.armature would not update the matrix cached by the
    # TSID formulation.
    gear_ratios = np.ones(robot.na)
    rotor_inertias = np.zeros(robot.na)
    for item in mapping:
        rotor_inertias[item["pin_v"]] = mj_model.dof_armature[item["mj_v"]]
    robot.set_gear_ratios(gear_ratios)
    robot.set_rotor_inertias(rotor_inertias)

    q_initial = pin.neutral(model)
    q_target = pin.neutral(model)
    for initial, target, item in zip(
        np.deg2rad(INITIAL_DEG), np.deg2rad(TARGET_DEG), mapping
    ):
        q_initial[item["pin_q"]] = initial
        q_target[item["pin_q"]] = target

    mj_data.qpos[:] = pin_q_to_mj(q_initial, mj_model, mapping)
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    formulation = tsid.InverseDynamicsFormulationAccForce(
        "ur5-tsid", robot, False
    )
    formulation.computeProblemData(0.0, q_initial, np.zeros(model.nv))

    posture = tsid.TaskJointPosture("posture", robot)
    posture.setKp(KP)
    posture.setKd(KD)
    posture.setMask(np.ones(robot.na))
    formulation.addMotionTask(posture, 1.0, 1, 0.0)

    reference = tsid.TrajectorySample(model.nq, model.nv)
    reference.value(q_target)
    reference.derivative(np.zeros(model.nv))
    reference.second_derivative(np.zeros(model.nv))
    posture.setReference(reference)

    solver = tsid.SolverHQuadProgFast("ur5-qpoases")
    solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

    dt = float(mj_model.opt.timestep)
    steps = int(np.ceil(args.duration / dt))
    print("========== 101 UR5 TSID + MuJoCo Posture Control ==========")
    print(f"Time step 控制周期: {dt * 1000.0:.3f} ms")
    print(f"Initial 初始角度: {INITIAL_DEG.tolist()} deg")
    print(f"Target 目标角度:  {TARGET_DEG.tolist()} deg")
    print(
        f"HQP dimensions: nVar={formulation.nVar}, "
        f"nEq={formulation.nEq}, nIn={formulation.nIn}"
    )
    print("Plant 被控对象: MuJoCo（非TSID内部积分）")

    solve_times_us = []
    saturation_steps = 0
    qp_failures = 0
    max_abs_ctrl = 0.0
    max_dynamic_residual = 0.0
    check_data = model.createData()

    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(mj_model, mj_data)
    else:
        viewer_context = nullcontext(None)

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = np.array([-0.15, 0.05, 0.75])
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -20.0

        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                print("Viewer closed early 可视化窗口提前关闭")
                break

            wall_start = time.perf_counter()
            q, v = mj_state_to_pin(mj_data, model, mapping)
            problem = formulation.computeProblemData(mj_data.time, q, v)

            solve_start = time.perf_counter()
            solution = solver.solve(problem)
            solve_times_us.append((time.perf_counter() - solve_start) * 1e6)

            if solution.status != 0:
                qp_failures += 1
                mj_data.ctrl[:] = 0.0
                print(
                    f"QP failed QP失败: step={step}, status={solution.status}"
                )
                break

            tau = np.asarray(
                formulation.getActuatorForces(solution)
            ).reshape(-1)
            ddq = np.asarray(
                formulation.getAccelerations(solution)
            ).reshape(-1)

            # Independent check of the TSID inverse-dynamics equality.  TSID
            # adds reflected rotor inertia separately from Pinocchio RNEA.
            tau_inverse_dynamics = (
                pin.rnea(model, check_data, q, v, ddq)
                + rotor_inertias * gear_ratios**2 * ddq
            )
            dynamic_residual = float(
                np.linalg.norm(tau - tau_inverse_dynamics)
            )
            max_dynamic_residual = max(max_dynamic_residual, dynamic_residual)

            ctrl, unclipped = generalized_tau_to_ctrl(
                tau, mj_data, mj_model, mapping
            )
            if np.max(np.abs(ctrl - unclipped)) > 1e-10:
                saturation_steps += 1
            max_abs_ctrl = max(max_abs_ctrl, float(np.max(np.abs(ctrl))))
            mj_data.ctrl[:] = ctrl
            mujoco.mj_step(mj_model, mj_data)

            if viewer is not None:
                viewer.sync()
                remaining = dt - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)

            if step % max(1, int(round(1.0 / dt))) == 0:
                q_now, v_now = mj_state_to_pin(mj_data, model, mapping)
                error_deg = np.rad2deg(pin.difference(model, q_now, q_target))
                print(
                    f"t={mj_data.time:5.2f}s | "
                    f"error={np.linalg.norm(error_deg):7.3f} deg | "
                    f"speed={np.linalg.norm(v_now):7.4f} rad/s | "
                    f"ctrl_max={np.max(np.abs(ctrl)):7.3f} Nm"
                )

    q_final, v_final = mj_state_to_pin(mj_data, model, mapping)
    final_error_deg_vector = np.rad2deg(
        pin.difference(model, q_final, q_target)
    )
    final_error_deg = float(np.linalg.norm(final_error_deg_vector))
    final_speed = float(np.linalg.norm(v_final))
    solve_times_us = np.asarray(solve_times_us)

    passed = (
        qp_failures == 0
        and saturation_steps == 0
        and final_error_deg < 0.05
        and final_speed < 0.01
        and max_dynamic_residual < 1e-8
    )

    print("\n========== 101 Result 结果 ==========")
    print(f"Final joint error 最终关节误差: {final_error_deg:.6f} deg")
    print(f"Per-joint error 各关节误差: {final_error_deg_vector.round(6).tolist()} deg")
    print(f"Final speed 最终速度范数: {final_speed:.6e} rad/s")
    print(f"Max actuator command 最大执行器指令: {max_abs_ctrl:.3f} Nm")
    print(f"Saturation steps 力矩饱和步数: {saturation_steps}")
    print(f"QP failures QP失败次数: {qp_failures}")
    print(f"Max dynamics residual 最大动力学残差: {max_dynamic_residual:.3e}")
    if solve_times_us.size:
        print(
            "QP solve time QP求解时间: "
            f"mean={np.mean(solve_times_us):.1f} us, "
            f"P95={np.percentile(solve_times_us, 95):.1f} us, "
            f"P99={np.percentile(solve_times_us, 99):.1f} us"
        )
    print(f"Result 结果: {'PASS 通过' if passed else 'FAIL 失败'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
