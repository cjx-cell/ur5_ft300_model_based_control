#!/usr/bin/env python3
"""91 - Validate UR5 + FT300 + inspection-probe MuJoCo/Pinocchio consistency."""

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "ur5_ft300_inspection.xml"
URDF_PATH = HERE / "ur5_ft300_inspection.urdf"
FRAME_NAME = "inspection_tip"


def mujoco_to_pinocchio_q(q_mj, mj_model, pin_model):
    q_pin = pin.neutral(pin_model)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if mj_joint_id < 0:
            raise RuntimeError(f"MuJoCo缺少关节: {name}")

        angle = q_mj[mj_model.jnt_qposadr[mj_joint_id]]
        qadr = pin_model.idx_qs[pin_joint_id]
        joint = pin_model.joints[pin_joint_id]

        if joint.nq == 1:
            q_pin[qadr] = angle
        elif joint.nq == 2 and joint.nv == 1:
            q_pin[qadr] = np.cos(angle)
            q_pin[qadr + 1] = np.sin(angle)
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


def pinocchio_vector_to_mujoco(x_pin, mj_model, pin_model):
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


def pinocchio_matrix_to_mujoco(M_pin, mj_model, pin_model):
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
    for path in [MJCF_PATH, URDF_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)

    pin_model = pin.buildModelFromUrdf(str(URDF_PATH))
    pin_data = pin_model.createData()

    if mj_model.nv != 6 or pin_model.nv != 6:
        raise RuntimeError(
            f"Unexpected nv: MuJoCo={mj_model.nv}, Pinocchio={pin_model.nv}"
        )

    frame_id = pin_model.getFrameId(FRAME_NAME)
    if frame_id >= pin_model.nframes:
        raise RuntimeError(f"Pinocchio找不到frame: {FRAME_NAME}")

    tip_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, FRAME_NAME
    )
    if tip_body_id < 0:
        raise RuntimeError(f"MuJoCo找不到body: {FRAME_NAME}")

    # Copy MuJoCo numerical damping / armature metadata into Pinocchio.
    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if mj_joint_id < 0:
            raise RuntimeError(f"MuJoCo缺少关节: {name}")

        pin_v = pin_model.idx_vs[pin_joint_id]
        mj_v = int(mj_model.jnt_dofadr[mj_joint_id])

        pin_model.armature[pin_v] = mj_model.dof_armature[mj_v]
        pin_model.damping[pin_v] = mj_model.dof_damping[mj_v]
        pin_model.friction[pin_v] = mj_model.dof_frictionloss[mj_v]

    rng = np.random.default_rng(91)
    sample_count = 10

    max_mass_error = 0.0
    max_bias_error = 0.0
    max_position_error = 0.0
    max_orientation_error = 0.0

    home_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if home_id < 0:
        raise RuntimeError("MuJoCo找不到home keyframe")

    for sample in range(sample_count):
        if sample == 0:
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
        M_pin_mj = pinocchio_matrix_to_mujoco(
            M_pin, mj_model, pin_model
        )

        M_mj = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, mj_data, M_mj)

        h_pin = pin.nonLinearEffects(
            pin_model, pin_data, q_pin, v_pin
        )
        h_pin_mj = pinocchio_vector_to_mujoco(
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

        pin_tip = pin_data.oMf[frame_id]
        mj_tip_position = mj_data.xpos[tip_body_id].copy()
        mj_tip_rotation = mj_data.xmat[tip_body_id].reshape(3, 3).copy()

        max_position_error = max(
            max_position_error,
            float(np.linalg.norm(pin_tip.translation - mj_tip_position)),
        )
        max_orientation_error = max(
            max_orientation_error,
            rotation_error_angle(pin_tip.rotation, mj_tip_rotation),
        )

    tolerance = 1e-6

    if (
        max_mass_error > tolerance
        or max_bias_error > tolerance
        or max_position_error > tolerance
        or max_orientation_error > tolerance
    ):
        raise RuntimeError(
            "UR5 model consistency failed UR5模型一致性失败: "
            f"M={max_mass_error:.3e}, "
            f"h={max_bias_error:.3e}, "
            f"p={max_position_error:.3e}, "
            f"R={max_orientation_error:.3e}"
        )

    print("========== 91 Validation 91 UR5模型验证 ==========")
    print(
        f"MuJoCo dimensions MuJoCo维度: "
        f"nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}"
    )
    print(
        f"Pinocchio dimensions Pinocchio维度: "
        f"nq={pin_model.nq}, nv={pin_model.nv}"
    )
    print(f"Mass matrix error 质量矩阵误差: {max_mass_error:.3e}")
    print(f"Nonlinear term error 非线性项误差: {max_bias_error:.3e}")
    print(f"TCP position error TCP位置误差: {max_position_error:.3e} m")
    print(
        f"TCP orientation error TCP姿态误差: "
        f"{np.degrees(max_orientation_error):.3e} deg"
    )
    print("Result 结果: PASS 通过")


if __name__ == "__main__":
    main()
