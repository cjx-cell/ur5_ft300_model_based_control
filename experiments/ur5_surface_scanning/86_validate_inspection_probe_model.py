#!/usr/bin/env python3
"""Validate the dedicated UR3 + FT300 inspection model."""

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "ur3_ft300_inspection.xml"
URDF_PATH = HERE / "ur3_ft300_inspection.urdf"

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def mujoco_to_pinocchio_q(q_mj, mj_model, pin_model):
    q_pin = pin.neutral(pin_model)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if mj_joint_id < 0:
            raise RuntimeError(f"MuJoCo模型缺少关节: {name}")

        angle = q_mj[mj_model.jnt_qposadr[mj_joint_id]]
        pin_q = pin_model.idx_qs[pin_joint_id]
        joint = pin_model.joints[pin_joint_id]

        if joint.nq == 1:
            q_pin[pin_q] = angle
        elif joint.nq == 2 and joint.nv == 1:
            q_pin[pin_q] = np.cos(angle)
            q_pin[pin_q + 1] = np.sin(angle)
        else:
            raise RuntimeError(f"不支持的关节配置表示: {name}")

    return q_pin


def mujoco_to_pinocchio_v(v_mj, mj_model, pin_model):
    v_pin = np.zeros(pin_model.nv)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        pin_v = pin_model.idx_vs[pin_joint_id]
        mj_v = int(mj_model.jnt_dofadr[mj_joint_id])
        v_pin[pin_v] = v_mj[mj_v]

    return v_pin


def pinocchio_to_mujoco_vector(x_pin, mj_model, pin_model):
    x_mj = np.zeros(mj_model.nv)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        x_mj[mj_model.jnt_dofadr[mj_joint_id]] = x_pin[
            pin_model.idx_vs[pin_joint_id]
        ]

    return x_mj


def pinocchio_to_mujoco_matrix(M_pin, mj_model, pin_model):
    pin_indices = np.empty(mj_model.nv, dtype=int)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        pin_indices[mj_model.jnt_dofadr[mj_joint_id]] = pin_model.idx_vs[
            pin_joint_id
        ]

    return np.asarray(M_pin)[np.ix_(pin_indices, pin_indices)]


def rotation_error_angle(R_a, R_b):
    R_error = R_a @ R_b.T
    cosine = np.clip((np.trace(R_error) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def main():
    if not MJCF_PATH.exists() or not URDF_PATH.exists():
        raise FileNotFoundError(
            "请先运行 86_build_inspection_probe_model.py 生成新模型"
        )

    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)

    pin_model = pin.buildModelFromUrdf(str(URDF_PATH))
    pin_data = pin_model.createData()

    if mj_model.nv != 6 or mj_model.nu != 6:
        raise RuntimeError(
            f"MuJoCo维度错误: nv={mj_model.nv}, nu={mj_model.nu}，预期均为6"
        )
    if pin_model.nv != 6:
        raise RuntimeError(f"Pinocchio维度错误: nv={pin_model.nv}，预期为6")

    for name in ARM_JOINTS:
        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            raise RuntimeError(f"缺少UR3关节: {name}")

    if mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_85_base_link"
    ) >= 0:
        raise RuntimeError("Robotiq夹爪仍存在于MuJoCo模型中")

    tip_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "inspection_tip"
    )
    if tip_body_id < 0:
        raise RuntimeError("MuJoCo中找不到 inspection_tip")

    tip_frame_id = pin_model.getFrameId("inspection_tip")
    if tip_frame_id >= len(pin_model.frames):
        raise RuntimeError("Pinocchio中找不到 inspection_tip")

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        pin_v = pin_model.idx_vs[pin_joint_id]
        mj_v = int(mj_model.jnt_dofadr[mj_joint_id])
        pin_model.armature[pin_v] = mj_model.dof_armature[mj_v]
        pin_model.damping[pin_v] = mj_model.dof_damping[mj_v]
        pin_model.friction[pin_v] = mj_model.dof_frictionloss[mj_v]

    rng = np.random.default_rng(86)
    sample_count = 10

    max_mass_error = 0.0
    max_bias_error = 0.0
    max_tip_position_error = 0.0
    max_tip_orientation_error = 0.0

    for sample in range(sample_count):
        if sample == 0:
            home_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_KEY, "home"
            )
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, home_id)
            mj_data.qvel[:] = 0.0
        else:
            mj_data.qpos[:] = rng.uniform(-1.0, 1.0, mj_model.nq)
            mj_data.qvel[:] = rng.uniform(-0.2, 0.2, mj_model.nv)

        mujoco.mj_forward(mj_model, mj_data)

        q_pin = mujoco_to_pinocchio_q(
            mj_data.qpos, mj_model, pin_model
        )
        v_pin = mujoco_to_pinocchio_v(
            mj_data.qvel, mj_model, pin_model
        )

        pin.forwardKinematics(pin_model, pin_data, q_pin, v_pin)
        pin.updateFramePlacements(pin_model, pin_data)

        M_pin = pin.crba(pin_model, pin_data, q_pin)
        M_pin_mj = pinocchio_to_mujoco_matrix(
            M_pin, mj_model, pin_model
        )

        M_mj = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, mj_data, M_mj)

        h_pin = pin.nonLinearEffects(pin_model, pin_data, q_pin, v_pin)
        h_pin_mj = pinocchio_to_mujoco_vector(
            h_pin, mj_model, pin_model
        )

        max_mass_error = max(
            max_mass_error,
            float(np.max(np.abs(M_mj - M_pin_mj))),
        )
        max_bias_error = max(
            max_bias_error,
            float(np.max(np.abs(mj_data.qfrc_bias - h_pin_mj))),
        )

        pin_tip = pin_data.oMf[tip_frame_id]
        mj_tip_position = mj_data.xpos[tip_body_id].copy()
        mj_tip_rotation = mj_data.xmat[tip_body_id].reshape(3, 3).copy()

        max_tip_position_error = max(
            max_tip_position_error,
            float(np.linalg.norm(pin_tip.translation - mj_tip_position)),
        )
        max_tip_orientation_error = max(
            max_tip_orientation_error,
            rotation_error_angle(pin_tip.rotation, mj_tip_rotation),
        )

    tolerance = 1e-6
    if (
        max_mass_error > tolerance
        or max_bias_error > tolerance
        or max_tip_position_error > tolerance
        or max_tip_orientation_error > tolerance
    ):
        raise RuntimeError(
            "模型一致性检查失败: "
            f"M={max_mass_error:.3e}, "
            f"h={max_bias_error:.3e}, "
            f"tip_pos={max_tip_position_error:.3e}, "
            f"tip_rot={max_tip_orientation_error:.3e}"
        )

    print("========== 86 Validation 86模型验证 ==========")
    print(f"MuJoCo dimensions MuJoCo维度: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    print(f"Pinocchio dimensions Pinocchio维度: nq={pin_model.nq}, nv={pin_model.nv}")
    print("Gripper removed 夹爪删除: PASS 通过")
    print("Inspection TCP 检测TCP: inspection_tip")
    print(f"Mass matrix error 质量矩阵误差: {max_mass_error:.3e}")
    print(f"Nonlinear term error 非线性项误差: {max_bias_error:.3e}")
    print(f"TCP position error TCP位置误差: {max_tip_position_error:.3e} m")
    print(f"TCP orientation error TCP姿态误差: {np.degrees(max_tip_orientation_error):.3e} deg")
    print("Result 结果: PASS 通过")


if __name__ == "__main__":
    main()
