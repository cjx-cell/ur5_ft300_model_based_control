#!/usr/bin/env python3
"""Build a UR3 + FT300 + inspection probe model from the validated Robotiq baseline.

The original validated model is kept unchanged. This script removes the Robotiq
2F-85 gripper and creates a dedicated inspection probe whose +Z axis is the tool
axis. The physical TCP is the distal point named inspection_tip.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
UR3_ARCHIVE = (
    HERE.parent
    / "archive"
    / "ur3_ft300_learning"
)

SOURCE_MJCF = (
    UR3_ARCHIVE
    / "ur3_ft300_robotiq_force_control.xml"
)
SOURCE_URDF = (
    UR3_ARCHIVE
    / "ur3_ft300_robotiq_force_control.urdf"
)

OUTPUT_MJCF = HERE / "ur3_ft300_inspection.xml"
OUTPUT_URDF = HERE / "ur3_ft300_inspection.urdf"

PROBE_LENGTH = 0.120
PROBE_RADIUS = 0.004
TIP_RADIUS = 0.005

PROBE_MASS = 0.120
PROBE_COM_Z = 0.06708333333333334
PROBE_IXX = 0.0001659125
PROBE_IYY = 0.0001659125
PROBE_IZZ = 0.000001

GRIPPER_BODY = "robotiq_85_base_link"
GRIPPER_ASSETS = {
    "robotiq_base",
    "left_knuckle",
    "right_knuckle",
    "left_finger",
    "right_finger",
    "left_inner_knuckle",
    "right_inner_knuckle",
}


def remove_child_by_name(parent: ET.Element, tag: str, name: str) -> bool:
    for child in list(parent):
        if child.tag == tag and child.attrib.get("name") == name:
            parent.remove(child)
            return True
    return False


def build_mjcf() -> None:
    tree = ET.parse(SOURCE_MJCF)
    root = tree.getroot()
    root.set("model", "ur3_ft300_inspection")

    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.attrib.get("name") in GRIPPER_ASSETS:
                asset.remove(mesh)

        ET.SubElement(
            asset,
            "material",
            name="inspection_probe_metal",
            rgba="0.70 0.72 0.75 1",
            specular="0.55",
            shininess="0.65",
        )
        ET.SubElement(
            asset,
            "material",
            name="inspection_tip_dark",
            rgba="0.10 0.12 0.14 1",
            specular="0.35",
            shininess="0.45",
        )

    ft_frame = root.find(".//body[@name='robotiq_ft_frame_id']")
    if ft_frame is None:
        raise RuntimeError("MJCF中找不到 robotiq_ft_frame_id")

    if not remove_child_by_name(ft_frame, "body", GRIPPER_BODY):
        raise RuntimeError("MJCF中找不到 Robotiq 夹爪根 body")

    probe = ET.SubElement(ft_frame, "body", name="inspection_probe")
    ET.SubElement(
        probe,
        "inertial",
        pos=f"0 0 {PROBE_COM_Z:.17g}",
        mass=f"{PROBE_MASS:.17g}",
        diaginertia=f"{PROBE_IXX:.17g} {PROBE_IYY:.17g} {PROBE_IZZ:.17g}",
    )

    # 杆身只用于显示，不参与环境接触；真正接触由球形检测头承担。
    ET.SubElement(
        probe,
        "geom",
        name="inspection_probe_shaft",
        type="cylinder",
        size=f"{PROBE_RADIUS:.17g} {(PROBE_LENGTH - TIP_RADIUS) / 2:.17g}",
        pos=f"0 0 {(PROBE_LENGTH - TIP_RADIUS) / 2:.17g}",
        material="inspection_probe_metal",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        probe,
        "geom",
        name="inspection_probe_tip_collision",
        type="sphere",
        size=f"{TIP_RADIUS:.17g}",
        pos=f"0 0 {PROBE_LENGTH - TIP_RADIUS:.17g}",
        material="inspection_tip_dark",
        friction="0.45 0.01 0.001",
        solref="-2500 -35",
    )

    tip = ET.SubElement(
        probe,
        "body",
        name="inspection_tip",
        pos=f"0 0 {PROBE_LENGTH:.17g}",
    )
    ET.SubElement(
        tip,
        "site",
        name="inspection_tip_site",
        pos="0 0 0",
        size="0.0035",
        rgba="1 0.15 0.05 0.9",
    )

    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)

    contact = root.find("contact")
    if contact is not None:
        for exclude in list(contact.findall("exclude")):
            body1 = exclude.attrib.get("body1", "")
            body2 = exclude.attrib.get("body2", "")
            if body1.startswith("robotiq_85_") or body2.startswith("robotiq_85_"):
                contact.remove(exclude)

    actuator = root.find("actuator")
    if actuator is not None:
        for child in list(actuator):
            if child.attrib.get("name") == "gripper_position":
                actuator.remove(child)

    keyframe = root.find("keyframe")
    if keyframe is not None:
        home = keyframe.find("key[@name='home']")
        if home is not None:
            old_qpos = home.attrib.get("qpos", "").split()
            old_ctrl = home.attrib.get("ctrl", "").split()
            home.set("qpos", " ".join(old_qpos[:6]))
            home.set("ctrl", " ".join(old_ctrl[:6]))

    root.insert(
        0,
        ET.Comment(
            " Dedicated UR3 + FT300 inspection model. "
            "Tool +Z points from the FT300 toward the inspection tip. "
        ),
    )

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_MJCF, encoding="utf-8", xml_declaration=True)


def add_urdf_probe(root: ET.Element) -> None:
    probe_link = ET.SubElement(root, "link", name="inspection_probe")

    inertial = ET.SubElement(probe_link, "inertial")
    ET.SubElement(inertial, "origin", xyz=f"0 0 {PROBE_COM_Z:.17g}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{PROBE_MASS:.17g}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{PROBE_IXX:.17g}",
        ixy="0",
        ixz="0",
        iyy=f"{PROBE_IYY:.17g}",
        iyz="0",
        izz=f"{PROBE_IZZ:.17g}",
    )

    visual_shaft = ET.SubElement(probe_link, "visual", name="inspection_probe_shaft")
    ET.SubElement(
        visual_shaft,
        "origin",
        xyz=f"0 0 {(PROBE_LENGTH - TIP_RADIUS) / 2:.17g}",
        rpy="0 0 0",
    )
    geometry = ET.SubElement(visual_shaft, "geometry")
    ET.SubElement(
        geometry,
        "cylinder",
        radius=f"{PROBE_RADIUS:.17g}",
        length=f"{PROBE_LENGTH - TIP_RADIUS:.17g}",
    )

    visual_tip = ET.SubElement(probe_link, "visual", name="inspection_probe_tip")
    ET.SubElement(
        visual_tip,
        "origin",
        xyz=f"0 0 {PROBE_LENGTH - TIP_RADIUS:.17g}",
        rpy="0 0 0",
    )
    geometry = ET.SubElement(visual_tip, "geometry")
    ET.SubElement(geometry, "sphere", radius=f"{TIP_RADIUS:.17g}")

    collision_tip = ET.SubElement(
        probe_link, "collision", name="inspection_probe_tip_collision"
    )
    ET.SubElement(
        collision_tip,
        "origin",
        xyz=f"0 0 {PROBE_LENGTH - TIP_RADIUS:.17g}",
        rpy="0 0 0",
    )
    geometry = ET.SubElement(collision_tip, "geometry")
    ET.SubElement(geometry, "sphere", radius=f"{TIP_RADIUS:.17g}")

    probe_joint = ET.SubElement(root, "joint", name="inspection_probe_joint", type="fixed")
    ET.SubElement(probe_joint, "parent", link="robotiq_ft_frame_id")
    ET.SubElement(probe_joint, "child", link="inspection_probe")
    ET.SubElement(probe_joint, "origin", xyz="0 0 0", rpy="0 0 0")

    ET.SubElement(root, "link", name="inspection_tip")
    tip_joint = ET.SubElement(root, "joint", name="inspection_tip_joint", type="fixed")
    ET.SubElement(tip_joint, "parent", link="inspection_probe")
    ET.SubElement(tip_joint, "child", link="inspection_tip")
    ET.SubElement(
        tip_joint,
        "origin",
        xyz=f"0 0 {PROBE_LENGTH:.17g}",
        rpy="0 0 0",
    )


def build_urdf() -> None:
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    root.set("name", "ur3_ft300_inspection")

    removed_links = {
        link.attrib["name"]
        for link in root.findall("link")
        if link.attrib.get("name", "").startswith("robotiq_85_")
    }

    for link in list(root.findall("link")):
        if link.attrib.get("name") in removed_links:
            root.remove(link)

    for joint in list(root.findall("joint")):
        name = joint.attrib.get("name", "")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib.get("link", "") if parent is not None else ""
        child_name = child.attrib.get("link", "") if child is not None else ""

        if (
            name.startswith("robotiq_85_")
            or parent_name in removed_links
            or child_name in removed_links
        ):
            root.remove(joint)

    add_urdf_probe(root)

    root.insert(
        0,
        ET.Comment(
            " UR3 + FT300 inspection-tool Pinocchio model. "
            "The inspection_tip frame is 120 mm along tool +Z from the FT300 frame. "
        ),
    )

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)


def main() -> None:
    build_mjcf()
    build_urdf()

    print("========== 86 Inspection Model 86检测工具模型 ==========")
    print(f"MuJoCo model MuJoCo模型: {OUTPUT_MJCF.name}")
    print(f"Pinocchio model Pinocchio模型: {OUTPUT_URDF.name}")
    print("Gripper removed 夹爪: removed 已删除")
    print(f"Probe length 检测工具长度: {PROBE_LENGTH * 1000:.1f} mm")
    print(f"Tip radius 检测头半径: {TIP_RADIUS * 1000:.1f} mm")
    print("Tool axis 工具轴: inspection +Z")
    print("TCP frame TCP坐标系: inspection_tip")


if __name__ == "__main__":
    main()
