#!/usr/bin/env python3
"""生成与项目 MuJoCo 模型对应的无相机 Pinocchio URDF。"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ur3_ft300_robotiq_mujoco/ur3_ft300_robotiq_resolved.urdf"
OUTPUT = HERE / "ur3_ft300_robotiq_force_control.urdf"
MJCF = HERE / "ur3_ft300_robotiq_force_control.xml"

JOINT_DAMPING = {
    "shoulder_pan_joint": 0.15,
    "shoulder_lift_joint": 0.15,
    "elbow_joint": 0.15,
    "wrist_1_joint": 0.08,
    "wrist_2_joint": 0.08,
    "wrist_3_joint": 0.08,
    "robotiq_85_left_knuckle_joint": 0.01,
    "robotiq_85_left_finger_tip_joint": 0.01,
    "robotiq_85_right_knuckle_joint": 0.01,
    "robotiq_85_right_finger_tip_joint": 0.01,
    "robotiq_85_left_inner_knuckle_joint": 0.01,
    "robotiq_85_right_inner_knuckle_joint": 0.01,
}


def is_camera_name(name: str) -> bool:
    return name.startswith("wrist_camera") or name.startswith("global_camera")


def synchronize_inertials_from_mujoco(root: ET.Element) -> None:
    """用当前 MJCF 的质量、质心和惯量覆盖 URDF，确保动力学一致。"""
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    for link in root.findall("link"):
        name = link.attrib["name"]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0 or model.body_mass[body_id] <= 0:
            continue

        old_inertial = link.find("inertial")
        if old_inertial is not None:
            link.remove(old_inertial)

        rotation_flat = np.empty(9)
        mujoco.mju_quat2Mat(rotation_flat, model.body_iquat[body_id])
        rotation = rotation_flat.reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T

        inertial = ET.Element("inertial")
        ET.SubElement(
            inertial,
            "origin",
            xyz=" ".join(f"{value:.17g}" for value in model.body_ipos[body_id]),
            rpy="0 0 0",
        )
        ET.SubElement(inertial, "mass", value=f"{model.body_mass[body_id]:.17g}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{inertia[0, 0]:.17g}",
            ixy=f"{inertia[0, 1]:.17g}",
            ixz=f"{inertia[0, 2]:.17g}",
            iyy=f"{inertia[1, 1]:.17g}",
            iyz=f"{inertia[1, 2]:.17g}",
            izz=f"{inertia[2, 2]:.17g}",
        )
        link.insert(0, inertial)


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("name", "ur3_ft300_robotiq_force_control")

    # 这是纯 Pinocchio URDF，不保留 MuJoCo 编译扩展。
    for extension in root.findall("mujoco"):
        root.remove(extension)

    removed_links = {
        link.attrib["name"]
        for link in root.findall("link")
        if is_camera_name(link.attrib["name"])
    }
    for link in list(root.findall("link")):
        if link.attrib["name"] in removed_links:
            root.remove(link)

    for joint in list(root.findall("joint")):
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib["link"] if parent is not None else ""
        child_name = child.attrib["link"] if child is not None else ""
        if parent_name in removed_links or child_name in removed_links:
            root.remove(joint)
            continue

        name = joint.attrib.get("name", "")
        if name in JOINT_DAMPING:
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", str(JOINT_DAMPING[name]))
            dynamics.set("friction", "0")

    synchronize_inertials_from_mujoco(root)

    root.insert(
        0,
        ET.Comment(
            " Camera-free Pinocchio model generated from the same resolved "
            "URDF used to create ur3_ft300_robotiq_force_control.xml. "
        ),
    )
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(f"已生成: {OUTPUT}")


if __name__ == "__main__":
    main()
