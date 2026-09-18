"""Shared, tested MuJoCo/TSID mapping helpers for experiments 101+.

The project keeps this module deliberately small: it owns model loading,
joint-state conversion, reflected rotor inertia, and direct-drive torque
mapping.  Task definitions remain visible in each numbered experiment.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import tsid


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class JointMap:
    name: str
    mj_joint: int
    mj_q: int
    mj_v: int
    pin_q: int
    pin_v: int
    actuator: int
    gear: float


@dataclass
class RobotSystem:
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    robot: object
    pin_model: pin.Model
    joints: list[JointMap]
    rotor_inertias: np.ndarray
    gear_ratios: np.ndarray


def load_robot_system(mjcf_path: Path, urdf_path: Path) -> RobotSystem:
    """Load matching models and configure TSID's reflected rotor inertia."""
    mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mj_data = mujoco.MjData(mj_model)

    package_dirs = pin.StdVec_StdString()
    package_dirs.append(str(urdf_path.parent))
    robot = tsid.RobotWrapper(str(urdf_path), package_dirs, False)
    pin_model = robot.model()

    if not (
        mj_model.nq == mj_model.nv == mj_model.nu == 6
        and pin_model.nq == pin_model.nv == int(robot.na) == 6
    ):
        raise RuntimeError("需要固定基、完全驱动的6自由度UR5模型")

    joints = []
    for actuator, name in enumerate(JOINT_NAMES):
        mj_joint = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        pin_joint = pin_model.getJointId(name)
        if mj_joint < 0 or pin_joint == 0:
            raise RuntimeError(f"关节映射失败: {name}")
        joint = pin_model.joints[pin_joint]
        transmission = int(mj_model.actuator_trnid[actuator, 0])
        gear = float(mj_model.actuator_gear[actuator, 0])
        if transmission != mj_joint or abs(gear) < 1e-12:
            raise RuntimeError(f"执行器不是预期的直接关节传动: {name}")
        joints.append(
            JointMap(
                name=name,
                mj_joint=mj_joint,
                mj_q=int(mj_model.jnt_qposadr[mj_joint]),
                mj_v=int(mj_model.jnt_dofadr[mj_joint]),
                pin_q=int(joint.idx_q),
                pin_v=int(joint.idx_v),
                actuator=actuator,
                gear=gear,
            )
        )

    gear_ratios = np.ones(robot.na)
    rotor_inertias = np.zeros(robot.na)
    for item in joints:
        # actuator gear is one in this MJCF.  Keeping both vectors explicit
        # makes the reflected inertia convention unambiguous.
        gear_ratios[item.pin_v] = item.gear
        rotor_inertias[item.pin_v] = mj_model.dof_armature[item.mj_v]
    robot.set_gear_ratios(gear_ratios)
    robot.set_rotor_inertias(rotor_inertias)

    return RobotSystem(
        mj_model=mj_model,
        mj_data=mj_data,
        robot=robot,
        pin_model=pin_model,
        joints=joints,
        rotor_inertias=rotor_inertias,
        gear_ratios=gear_ratios,
    )


def configuration_from_degrees(system: RobotSystem, values_deg):
    q = pin.neutral(system.pin_model)
    for value, item in zip(np.deg2rad(values_deg), system.joints):
        q[item.pin_q] = value
    return q


def reset_mujoco(system: RobotSystem, q_pin):
    system.mj_data.qpos[:] = 0.0
    system.mj_data.qvel[:] = 0.0
    for item in system.joints:
        system.mj_data.qpos[item.mj_q] = q_pin[item.pin_q]
    mujoco.mj_forward(system.mj_model, system.mj_data)


def read_state(system: RobotSystem):
    q = pin.neutral(system.pin_model)
    v = np.zeros(system.pin_model.nv)
    for item in system.joints:
        q[item.pin_q] = system.mj_data.qpos[item.mj_q]
        v[item.pin_v] = system.mj_data.qvel[item.mj_v]
    return q, v


def actuator_bounds(system: RobotSystem):
    lower = np.full(system.robot.na, -np.inf)
    upper = np.full(system.robot.na, np.inf)
    for item in system.joints:
        if system.mj_model.actuator_ctrllimited[item.actuator]:
            low, high = system.mj_model.actuator_ctrlrange[item.actuator]
            lower[item.pin_v] = low * item.gear
            upper[item.pin_v] = high * item.gear
    return lower, upper


def tau_to_ctrl(system: RobotSystem, tau, *, clip=False):
    """Convert TSID generalized actuation to MuJoCo motor control.

    The returned violation is the largest amount by which an unclipped command
    exceeds an actuator control range.  Constrained experiments leave
    ``clip=False`` so any HQP mistake remains observable.
    """
    ctrl = np.zeros(system.mj_model.nu)
    max_violation = 0.0
    for item in system.joints:
        requested = (
            tau[item.pin_v] - system.mj_data.qfrc_passive[item.mj_v]
        ) / item.gear
        if system.mj_model.actuator_ctrllimited[item.actuator]:
            low, high = system.mj_model.actuator_ctrlrange[item.actuator]
            max_violation = max(
                max_violation,
                float(max(low - requested, requested - high, 0.0)),
            )
            if clip:
                requested = float(np.clip(requested, low, high))
        ctrl[item.actuator] = requested
    return ctrl, max_violation


def inverse_dynamics_residual(system: RobotSystem, data, q, v, ddq, tau):
    expected = (
        pin.rnea(system.pin_model, data, q, v, ddq)
        + system.rotor_inertias * system.gear_ratios**2 * ddq
    )
    return float(np.linalg.norm(tau - expected))


def frame_pose(system: RobotSystem, data, frame_id, q):
    pin.forwardKinematics(system.pin_model, data, q)
    pin.updateFramePlacements(system.pin_model, data)
    pose = data.oMf[frame_id]
    return pin.SE3(pose.rotation.copy(), pose.translation.copy())


def se3_sample(pose: pin.SE3):
    sample = tsid.TrajectorySample(12, 6)
    sample.value(np.asarray(tsid.SE3ToVector(pose)).reshape(12))
    sample.derivative(np.zeros(6))
    sample.second_derivative(np.zeros(6))
    return sample


def posture_sample(model: pin.Model, q):
    sample = tsid.TrajectorySample(model.nq, model.nv)
    sample.value(q)
    sample.derivative(np.zeros(model.nv))
    sample.second_derivative(np.zeros(model.nv))
    return sample
