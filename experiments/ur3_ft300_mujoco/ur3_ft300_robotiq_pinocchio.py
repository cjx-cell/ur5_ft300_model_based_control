#!/usr/bin/env python3
"""加载与项目 MJCF 匹配的 Pinocchio 模型，并提供状态/力矩映射。"""

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "ur3_ft300_robotiq_force_control.urdf"
MJCF_PATH = HERE / "ur3_ft300_robotiq_force_control.xml"


def load_models():
    """加载两个模型，并把 MuJoCo armature/damping 同步到 Pinocchio。"""
    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    pin_model = pin.buildModelFromUrdf(str(URDF_PATH))

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if mj_joint_id < 0:
            raise RuntimeError(f"MuJoCo 模型缺少关节: {name}")

        pin_v = pin_model.idx_vs[pin_joint_id]
        mj_v = int(mj_model.jnt_dofadr[mj_joint_id])
        if pin_model.joints[pin_joint_id].nv != 1:
            raise RuntimeError(f"目前只支持单自由度关节: {name}")

        pin_model.armature[pin_v] = mj_model.dof_armature[mj_v]
        pin_model.damping[pin_v] = mj_model.dof_damping[mj_v]
        pin_model.friction[pin_v] = mj_model.dof_frictionloss[mj_v]

    return mj_model, pin_model


def mujoco_to_pinocchio_q(q_mj, mj_model, pin_model):
    """按关节名称转换 q；自动处理 Pinocchio 连续关节的 cos/sin 表示。"""
    q_mj = np.asarray(q_mj)
    q_pin = pin.neutral(pin_model)

    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        mj_q = int(mj_model.jnt_qposadr[mj_joint_id])
        pin_q = pin_model.idx_qs[pin_joint_id]
        joint = pin_model.joints[pin_joint_id]
        angle = q_mj[mj_q]

        if joint.nq == 1:
            q_pin[pin_q] = angle
        elif joint.nq == 2 and joint.nv == 1:
            q_pin[pin_q] = np.cos(angle)
            q_pin[pin_q + 1] = np.sin(angle)
        else:
            raise RuntimeError(f"不支持的关节配置表示: {name}")

    return q_pin


def mujoco_to_pinocchio_v(v_mj, mj_model, pin_model):
    """按关节名称把 MuJoCo 广义速度转换为 Pinocchio 顺序。"""
    v_mj = np.asarray(v_mj)
    v_pin = np.zeros(pin_model.nv)
    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        v_pin[pin_model.idx_vs[pin_joint_id]] = v_mj[
            mj_model.jnt_dofadr[mj_joint_id]
        ]
    return v_pin


def pinocchio_to_mujoco_tau(tau_pin, mj_model, pin_model):
    """按关节名称把 Pinocchio 广义力转换为 MuJoCo 顺序。"""
    tau_pin = np.asarray(tau_pin)
    tau_mj = np.zeros(mj_model.nv)
    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        tau_mj[mj_model.jnt_dofadr[mj_joint_id]] = tau_pin[
            pin_model.idx_vs[pin_joint_id]
        ]
    return tau_mj


def pinocchio_matrix_to_mujoco(matrix_pin, mj_model, pin_model):
    """将 Pinocchio nv×nv 矩阵排列成 MuJoCo 自由度顺序。"""
    pin_indices = np.empty(mj_model.nv, dtype=int)
    for pin_joint_id in range(1, pin_model.njoints):
        name = pin_model.names[pin_joint_id]
        mj_joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        pin_indices[mj_model.jnt_dofadr[mj_joint_id]] = pin_model.idx_vs[
            pin_joint_id
        ]
    return np.asarray(matrix_pin)[np.ix_(pin_indices, pin_indices)]


def passive_torque_pinocchio(v_pin, pin_model):
    """返回与 MuJoCo joint damping/friction 对应的被动力矩。"""
    v_pin = np.asarray(v_pin)
    return -pin_model.damping * v_pin - pin_model.friction * np.sign(v_pin)


def validate_consistency(sample_count=10):
    """在 home 和随机姿态下比较质量矩阵及重力/科氏项。"""
    mj_model, pin_model = load_models()
    mj_data = mujoco.MjData(mj_model)
    pin_data = pin_model.createData()
    rng = np.random.default_rng(7)

    max_mass_error = 0.0
    max_bias_error = 0.0
    for sample in range(sample_count):
        if sample == 0:
            key_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_KEY, "home"
            )
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        else:
            mj_data.qpos[:6] = rng.uniform(-1.0, 1.0, 6)
            master = rng.uniform(0.0, 0.6)
            gripper_positions = {
                "robotiq_85_left_knuckle_joint": master,
                "robotiq_85_left_finger_tip_joint": -master,
                "robotiq_85_right_knuckle_joint": -master,
                "robotiq_85_right_finger_tip_joint": master,
                "robotiq_85_left_inner_knuckle_joint": master,
                "robotiq_85_right_inner_knuckle_joint": -master,
            }
            for name, value in gripper_positions.items():
                joint_id = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                mj_data.qpos[mj_model.jnt_qposadr[joint_id]] = value
            mj_data.qvel[:] = rng.uniform(-0.2, 0.2, mj_model.nv)

        mujoco.mj_forward(mj_model, mj_data)
        q_pin = mujoco_to_pinocchio_q(
            mj_data.qpos, mj_model, pin_model
        )
        v_pin = mujoco_to_pinocchio_v(
            mj_data.qvel, mj_model, pin_model
        )

        mass_pin = pin.crba(pin_model, pin_data, q_pin)
        mass_pin_mj_order = pinocchio_matrix_to_mujoco(
            mass_pin, mj_model, pin_model
        )
        mass_mj = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, mj_data, mass_mj)

        bias_pin = pin.nonLinearEffects(pin_model, pin_data, q_pin, v_pin)
        bias_pin_mj_order = pinocchio_to_mujoco_tau(
            bias_pin, mj_model, pin_model
        )

        max_mass_error = max(
            max_mass_error,
            float(np.max(np.abs(mass_mj - mass_pin_mj_order))),
        )
        max_bias_error = max(
            max_bias_error,
            float(np.max(np.abs(mj_data.qfrc_bias - bias_pin_mj_order))),
        )

    tolerance = 1e-6
    if max_mass_error > tolerance or max_bias_error > tolerance:
        raise RuntimeError(
            "模型不一致: "
            f"质量矩阵误差={max_mass_error:.3e}, "
            f"非线性项误差={max_bias_error:.3e}"
        )
    print(f"通过 {sample_count} 组状态的一致性检查")
    print(f"质量矩阵最大绝对误差: {max_mass_error:.3e}")
    print(f"重力/科氏项最大绝对误差: {max_bias_error:.3e}")


if __name__ == "__main__":
    validate_consistency()
