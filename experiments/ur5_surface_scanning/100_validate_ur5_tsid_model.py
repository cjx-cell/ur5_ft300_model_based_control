#!/usr/bin/env python3
"""100 - Validate the UR5 model shared by MuJoCo, Pinocchio and TSID.

This is the gate before closing a TSID torque loop around MuJoCo.  It checks
joint/actuator mappings, TCP forward kinematics, mass matrices and gravity
torques at several representative configurations.  No viewer is opened.
"""

from pathlib import Path
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

FRAME_NAME = "inspection_tip"
SITE_NAME = "inspection_tip_site"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# These tolerances are deliberately much tighter than control accuracy needs.
POSITION_TOL = 2.0e-6       # m
ORIENTATION_TOL = 2.0e-6    # rad
MASS_MATRIX_TOL = 2.0e-4    # relative Frobenius error
GRAVITY_TOL = 2.0e-4        # relative 2-norm error


def pin_indices(model, joint_name):
    joint_id = model.getJointId(joint_name)
    if joint_id == 0:
        raise RuntimeError(f"Pinocchio找不到关节: {joint_name}")
    joint = model.joints[joint_id]
    if joint.nq != 1 or joint.nv != 1:
        raise RuntimeError(f"{joint_name} 不是标量关节")
    return int(joint.idx_q), int(joint.idx_v)


def mappings(mj_model, pin_model):
    rows = []
    for name in JOINT_NAMES:
        mj_joint = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if mj_joint < 0:
            raise RuntimeError(f"MuJoCo找不到关节: {name}")
        pin_q, pin_v = pin_indices(pin_model, name)
        rows.append(
            (
                name,
                int(mj_model.jnt_qposadr[mj_joint]),
                int(mj_model.jnt_dofadr[mj_joint]),
                pin_q,
                pin_v,
                mj_joint,
            )
        )
    return rows


def arm_q_to_models(q_arm, rows, mj_model, pin_model):
    q_mj = np.zeros(mj_model.nq)
    q_pin = pin.neutral(pin_model)
    for value, (_, mj_q, _, pin_q, _, _) in zip(q_arm, rows):
        q_mj[mj_q] = value
        q_pin[pin_q] = value
    return q_mj, q_pin


def sync_armature(pin_model, mj_model, rows):
    for _, _, mj_v, _, pin_v, _ in rows:
        pin_model.armature[pin_v] = mj_model.dof_armature[mj_v]


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-12))


def main():
    for path in (MJCF_PATH, URDF_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)

    package_dirs = pin.StdVec_StdString()
    package_dirs.append(str(URDF_PATH.parent))
    robot = tsid.RobotWrapper(str(URDF_PATH), package_dirs, False)
    pin_model = robot.model()
    pin_data = pin_model.createData()

    rows = mappings(mj_model, pin_model)
    sync_armature(pin_model, mj_model, rows)

    frame_id = pin_model.getFrameId(FRAME_NAME)
    site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME
    )
    if frame_id >= pin_model.nframes or site_id < 0:
        raise RuntimeError("inspection_tip frame/site 不完整")

    print("========== 100 UR5 Model Validation 模型一致性验证 ==========")
    print(
        f"Dimensions 维度: MuJoCo nq/nv/nu={mj_model.nq}/{mj_model.nv}/{mj_model.nu}, "
        f"Pinocchio nq/nv={pin_model.nq}/{pin_model.nv}, TSID na={robot.na}"
    )

    passed = (
        mj_model.nq == mj_model.nv == mj_model.nu == 6
        and pin_model.nq == pin_model.nv == int(robot.na) == 6
    )

    print("\nJoint and actuator mapping 关节与执行器映射:")
    for actuator_id, row in enumerate(rows):
        name, mj_q, mj_v, pin_q, pin_v, mj_joint = row
        transmission_joint = int(mj_model.actuator_trnid[actuator_id, 0])
        gear = float(mj_model.actuator_gear[actuator_id, 0])
        actuator_name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        mapping_ok = transmission_joint == mj_joint and abs(gear - 1.0) < 1e-12
        passed &= mapping_ok
        print(
            f"  {actuator_id}: {actuator_name} -> {name} | "
            f"mj(q={mj_q},v={mj_v}) pin(q={pin_q},v={pin_v}) "
            f"gear={gear:g} [{'OK' if mapping_ok else 'ERROR'}]"
        )

    test_configurations_deg = np.array(
        [
            [0, -60, 90, -30, 45, 0],
            [-72, -90, 90, -90, -90, 0],
            [25, -70, 105, -55, -65, 30],
            [-35, -110, 80, -45, -80, -20],
            [55, -50, 65, -100, -45, 70],
        ],
        dtype=float,
    )

    max_position_error = 0.0
    max_orientation_error = 0.0
    max_mass_error = 0.0
    max_gravity_error = 0.0

    print("\nCross-engine checks 跨引擎检查:")
    for index, q_deg in enumerate(test_configurations_deg, start=1):
        q_mj, q_pin = arm_q_to_models(
            np.deg2rad(q_deg), rows, mj_model, pin_model
        )
        mj_data.qpos[:] = q_mj
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        pin.forwardKinematics(pin_model, pin_data, q_pin)
        pin.updateFramePlacements(pin_model, pin_data)
        placement = pin_data.oMf[frame_id]

        position_error = float(
            np.linalg.norm(mj_data.site_xpos[site_id] - placement.translation)
        )
        R_mj = mj_data.site_xmat[site_id].reshape(3, 3)
        orientation_error = float(
            np.linalg.norm(pin.log3(placement.rotation.T @ R_mj))
        )

        mass_mj = np.zeros((mj_model.nv, mj_model.nv))
        # MuJoCo 3.13 takes MjData directly; older Python bindings accepted
        # the packed ``qM`` array as the third argument.
        try:
            mujoco.mj_fullM(mj_model, mj_data, mass_mj)
        except TypeError:
            packed_mass = getattr(mj_data, "qM", mj_data.M)
            mujoco.mj_fullM(mj_model, mass_mj, packed_mass)
        mass_pin = pin.crba(pin_model, pin_data, q_pin)
        mass_pin = 0.5 * (mass_pin + mass_pin.T)
        mass_error = relative_error(mass_mj, mass_pin)

        gravity_pin = pin.computeGeneralizedGravity(pin_model, pin_data, q_pin)
        gravity_error = relative_error(mj_data.qfrc_bias, gravity_pin)

        max_position_error = max(max_position_error, position_error)
        max_orientation_error = max(max_orientation_error, orientation_error)
        max_mass_error = max(max_mass_error, mass_error)
        max_gravity_error = max(max_gravity_error, gravity_error)

        print(
            f"  pose {index}: pos={position_error:.3e} m, "
            f"ori={orientation_error:.3e} rad, "
            f"Mrel={mass_error:.3e}, grel={gravity_error:.3e}"
        )

    passed &= max_position_error <= POSITION_TOL
    passed &= max_orientation_error <= ORIENTATION_TOL
    passed &= max_mass_error <= MASS_MATRIX_TOL
    passed &= max_gravity_error <= GRAVITY_TOL

    print("\n========== 100 Result 结果 ==========")
    print(f"Max TCP position mismatch 最大位置差: {max_position_error:.3e} m")
    print(f"Max TCP orientation mismatch 最大姿态差: {max_orientation_error:.3e} rad")
    print(f"Max mass-matrix relative error 最大质量矩阵相对误差: {max_mass_error:.3e}")
    print(f"Max gravity relative error 最大重力相对误差: {max_gravity_error:.3e}")
    print(f"Result 结果: {'PASS 通过' if passed else 'FAIL 失败'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
