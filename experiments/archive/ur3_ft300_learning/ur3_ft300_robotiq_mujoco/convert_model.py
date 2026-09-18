#!/usr/bin/env python3
"""Convert the Gazebo UR3 + FT300 + Robotiq xacro into a usable MJCF model.

The source ROS files are read-only inputs.  All generated files are written next
to this script.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import xml.etree.ElementTree as ET

import mujoco


HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/home/ubuntu/ur3_ft300_ws")
SOURCE_XACRO = (
    WORKSPACE
    / "src/ur_simulation_gz/ur_simulation_gz/urdf"
    / "ur3_ft300_robotiq_2f85.urdf.xacro"
)

RESOLVED_URDF = HERE / "ur3_ft300_robotiq_resolved.urdf"
RAW_MJCF = HERE / "ur3_ft300_robotiq_raw.xml"
FINAL_MJCF = HERE / "ur3_ft300_robotiq.xml"

PACKAGE_PATHS = {
    "ur_description": WORKSPACE / "install/ur_description/share/ur_description",
    "robotiq_description": (
        WORKSPACE / "install/robotiq_description/share/robotiq_description"
    ),
    "robotiq_ft_sensor_description": (
        WORKSPACE
        / "install/robotiq_ft_sensor_description/share"
        / "robotiq_ft_sensor_description"
    ),
    "realsense2_description": Path(
        "/opt/ros/humble/share/realsense2_description"
    ),
}

ARM_ACTUATORS = (
    ("shoulder_pan_motor", "shoulder_pan_joint", 56.0),
    ("shoulder_lift_motor", "shoulder_lift_joint", 56.0),
    ("elbow_motor", "elbow_joint", 28.0),
    ("wrist1_motor", "wrist_1_joint", 12.0),
    ("wrist2_motor", "wrist_2_joint", 12.0),
    ("wrist3_motor", "wrist_3_joint", 12.0),
)

ARM_JOINTS = tuple(item[1] for item in ARM_ACTUATORS)
GRIPPER_JOINTS = (
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_right_finger_tip_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
)

# Same arm home configuration as the Gazebo ros2_control block.  The last six
# entries are the Robotiq master and mimic joints in MuJoCo qpos order.
HOME_QPOS = (
    -1.254,
    -1.5707,
    1.5707,
    -1.5707,
    -1.5707,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def expand_xacro() -> str:
    xacro_args = [
        "xacro",
        str(SOURCE_XACRO),
        "name:=ur",
        "ur_type:=ur3",
        "tf_prefix:=",
        "safety_limits:=true",
        "safety_pos_margin:=0.15",
        "safety_k_position:=20",
        "sim_ignition:=false",
        "sim_position_gain:=0.5",
        "use_fake_hardware:=true",
        "gripper_use_fake_hardware:=true",
        "ft_sensor_use_fake_mode:=true",
    ]
    command = " && ".join(
        (
            "source /opt/ros/humble/setup.bash",
            f"source {shlex.quote(str(WORKSPACE / 'install/setup.bash'))}",
            shlex.join(xacro_args),
        )
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def resolve_mesh_uri(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[len("file://") :]
    if not uri.startswith("package://"):
        return uri

    package_and_path = uri[len("package://") :]
    package, relative = package_and_path.split("/", 1)
    if package not in PACKAGE_PATHS:
        raise RuntimeError(f"No package path configured for {package!r}")
    resolved = PACKAGE_PATHS[package] / relative
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return str(resolved.resolve())


def make_resolved_urdf(xacro_xml: str) -> None:
    root = ET.fromstring(xacro_xml)

    # Gazebo and ros2_control elements are not MuJoCo definitions.  Their
    # useful parts are recreated explicitly in MJCF below.
    for child in list(root):
        if child.tag in {"gazebo", "ros2_control"}:
            root.remove(child)

    for mesh in root.findall(".//mesh"):
        mesh.set("filename", resolve_mesh_uri(mesh.attrib["filename"]))

    mujoco_extension = ET.Element("mujoco")
    ET.SubElement(
        mujoco_extension,
        "compiler",
        {
            # The installed MuJoCo build does not decode the Collada visual
            # meshes.  Collision STL/primitive geometry remains visible.
            "discardvisual": "true",
            # Keep fixed FT and camera frames as named bodies.
            "fusestatic": "false",
            "balanceinertia": "true",
        },
    )
    root.insert(0, mujoco_extension)

    ET.indent(root, space="  ")
    RESOLVED_URDF.write_text(
        ET.tostring(root, encoding="unicode", xml_declaration=True),
        encoding="utf-8",
    )


def find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"Converted MJCF has no body named {name!r}")
    return body


def configure_joint(joint: ET.Element, damping: float, armature: float) -> None:
    joint.set("damping", f"{damping:g}")
    joint.set("armature", f"{armature:g}")


def enhance_mjcf() -> None:
    model = mujoco.MjModel.from_xml_path(str(RESOLVED_URDF))
    mujoco.mj_saveLastXML(str(RAW_MJCF), model)

    tree = ET.parse(RAW_MJCF)
    root = tree.getroot()
    root.set("model", "ur3_ft300_robotiq")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.001")
    option.set("integrator", "implicitfast")
    option.set("gravity", "0 0 -9.81")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")

    # Add modest numerical damping/armature without changing the URDF inertia.
    for joint_name in ARM_JOINTS:
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise RuntimeError(f"Missing arm joint {joint_name}")
        wrist = joint_name.startswith("wrist")
        configure_joint(joint, 0.08 if wrist else 0.15, 0.001 if wrist else 0.01)

    for joint_name in GRIPPER_JOINTS:
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise RuntimeError(f"Missing gripper joint {joint_name}")
        configure_joint(joint, 0.01, 0.0001)

    # URDF mimic joints become soft equality constraints.  Tighten them enough
    # for the closed-chain gripper while keeping the 1 ms simulation stable.
    equality = root.find("equality")
    if equality is None or len(equality) != 5:
        raise RuntimeError("Expected five imported Robotiq mimic constraints")
    for constraint in equality:
        constraint.set("solref", "0.002 1")
        constraint.set("solimp", "0.99 0.999 0.001")

    # Assign physical contact parameters and simple display colors to the
    # imported collision geometry.
    for body in root.findall(".//body"):
        body_name = body.attrib.get("name", "")
        for geom in body.findall("geom"):
            geom.set("friction", "0.8 0.01 0.001")
            if body_name.startswith("robotiq_85"):
                geom.set("rgba", "0.18 0.18 0.20 1")
            elif body_name.startswith("ft300") or body_name == "robotiq_ft_frame_id":
                geom.set("rgba", "0.12 0.12 0.12 1")
            else:
                geom.set("rgba", "0.15 0.45 0.75 1")

            if geom.attrib.get("name") in {
                "left_rubber_pad_collision",
                "right_rubber_pad_collision",
            }:
                geom.set("friction", "1.2 0.01 0.001")
                geom.set("solref", "-2000 -20")
                geom.set("rgba", "0.05 0.05 0.05 1")

    # Keep camera frames and provide lightweight visible housings.  The camera
    # optical frames themselves come from the original Realsense xacros.
    for body_name, rgba in (
        ("wrist_camera_link", "0.08 0.08 0.08 1"),
        ("global_camera_link", "0.12 0.12 0.12 1"),
    ):
        ET.SubElement(
            find_body(root, body_name),
            "geom",
            name=f"{body_name}_housing",
            type="box",
            size="0.045 0.012 0.012",
            rgba=rgba,
            contype="0",
            conaffinity="0",
            group="2",
        )

    # ROS optical frame: +X right, +Y down, +Z forward.  MuJoCo cameras look
    # along -Z with +Y up, hence the 180 degree rotation around local X.
    ET.SubElement(
        find_body(root, "wrist_camera_color_optical_frame"),
        "camera",
        name="wrist_camera",
        pos="0 0 0",
        quat="0 1 0 0",
        fovy="46.8",
    )
    ET.SubElement(
        find_body(root, "global_camera_color_optical_frame"),
        "camera",
        name="global_camera",
        pos="0 0 0",
        quat="0 1 0 0",
        fovy="55.5",
    )

    # A force/torque sensor measures the wrench transmitted between the body
    # containing this site and its parent.  Keeping static bodies unfused is
    # therefore essential.
    ET.SubElement(
        find_body(root, "ft300_sensor"),
        "site",
        name="ft300_site",
        pos="0 0 0",
        size="0.006",
        rgba="1 0.2 0.1 0.8",
    )

    contact = ET.SubElement(root, "contact")
    for body1, body2 in (
        ("base_link_inertia", "shoulder_link"),
        (
            "robotiq_85_left_finger_tip_link",
            "robotiq_85_left_inner_knuckle_link",
        ),
        (
            "robotiq_85_right_finger_tip_link",
            "robotiq_85_right_inner_knuckle_link",
        ),
    ):
        ET.SubElement(contact, "exclude", body1=body1, body2=body2)

    actuator = ET.SubElement(root, "actuator")
    for actuator_name, joint_name, limit in ARM_ACTUATORS:
        ET.SubElement(
            actuator,
            "motor",
            name=actuator_name,
            joint=joint_name,
            gear="1",
            ctrllimited="true",
            ctrlrange=f"{-limit:g} {limit:g}",
        )

    # ctrl[6] is a desired opening angle in radians; only the master knuckle is
    # actuated and the five imported equality constraints close the mechanism.
    ET.SubElement(
        actuator,
        "position",
        name="gripper_position",
        joint="robotiq_85_left_knuckle_joint",
        kp="50",
        ctrllimited="true",
        ctrlrange="-0.01 0.8",
        forcelimited="true",
        forcerange="-50 50",
    )

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", name="ft300_force", site="ft300_site")
    ET.SubElement(sensor, "torque", name="ft300_torque", site="ft300_site")

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="home",
        qpos=" ".join(f"{value:g}" for value in HOME_QPOS),
        ctrl="0 0 0 0 0 0 0",
    )

    ET.indent(tree, space="  ")
    tree.write(FINAL_MJCF, encoding="unicode", xml_declaration=True)


def validate() -> None:
    model = mujoco.MjModel.from_xml_path(str(FINAL_MJCF))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    expected = {
        "nq": 12,
        "nv": 12,
        "nu": 7,
        "neq": 5,
        "nsensor": 2,
        "ncam": 2,
    }
    actual = {name: int(getattr(model, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected model dimensions: {actual}, expected {expected}")
    if data.ncon:
        contacts = [
            (
                model.body(model.geom_bodyid[c.geom1]).name,
                model.body(model.geom_bodyid[c.geom2]).name,
                float(c.dist),
            )
            for c in data.contact
        ]
        raise RuntimeError(f"Home pose has unexpected contacts: {contacts}")

    print(f"Wrote: {RESOLVED_URDF}")
    print(f"Wrote: {RAW_MJCF}")
    print(f"Wrote: {FINAL_MJCF}")
    print("Validation passed:", actual, "home_contacts=0")


def main() -> None:
    os.chdir(HERE)
    make_resolved_urdf(expand_xacro())
    enhance_mjcf()
    validate()


if __name__ == "__main__":
    main()
