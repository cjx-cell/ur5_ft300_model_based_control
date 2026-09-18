#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/home/ubuntu/ur3_ft300_ws")
UR_DESCRIPTION = WORKSPACE / "install/ur_description/share/ur_description"
FT_DESCRIPTION = (
    WORKSPACE / "install/robotiq_ft_sensor_description/share"
    / "robotiq_ft_sensor_description"
)

SOURCE_XACRO = UR_DESCRIPTION / "urdf/ur.urdf.xacro"

RESOLVED_URDF = HERE / "ur5_ft300_inspection_resolved.urdf"
RAW_MJCF = HERE / "ur5_ft300_inspection_raw.xml"
FINAL_MJCF = HERE / "ur5_ft300_inspection.xml"
PIN_URDF = HERE / "ur5_ft300_inspection.urdf"

ROBOT_BASE_X = -0.200
ROBOT_BASE_Y = 0.050
ROBOT_BASE_Z = 0.760

PROBE_LENGTH = 0.120
PROBE_RADIUS = 0.004
TIP_RADIUS = 0.005

PROBE_MASS = 0.120
PROBE_COM_Z = 0.06708333333333334
PROBE_IXX = 0.0001659125
PROBE_IYY = 0.0001659125
PROBE_IZZ = 0.000001

HOME_Q = np.array([
    -1.2540,
    -1.5707,
    +1.5707,
    -1.5707,
    -1.5707,
    0.0,
], dtype=float)

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ACTUATOR_NAMES = [
    "shoulder_pan_motor",
    "shoulder_lift_motor",
    "elbow_motor",
    "wrist1_motor",
    "wrist2_motor",
    "wrist3_motor",
]

JOINT_DAMPING = {
    "shoulder_pan_joint": 0.15,
    "shoulder_lift_joint": 0.15,
    "elbow_joint": 0.15,
    "wrist_1_joint": 0.08,
    "wrist_2_joint": 0.08,
    "wrist_3_joint": 0.08,
}

JOINT_ARMATURE = {
    "shoulder_pan_joint": 0.01,
    "shoulder_lift_joint": 0.01,
    "elbow_joint": 0.01,
    "wrist_1_joint": 0.001,
    "wrist_2_joint": 0.001,
    "wrist_3_joint": 0.001,
}


def require_paths():
    required = [
        WORKSPACE / "install/setup.bash",
        SOURCE_XACRO,
        FT_DESCRIPTION,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required ROS files 缺少ROS模型文件:\n"
            + "\n".join(str(p) for p in missing)
        )


def expand_ur5_xacro():
    args = [
        "xacro",
        str(SOURCE_XACRO),
        "name:=ur5",
        "ur_type:=ur5",
        "tf_prefix:=",
        "safety_limits:=false",
        "sim_gazebo:=false",
        "sim_ignition:=false",
        "use_fake_hardware:=true",
    ]
    command = " && ".join([
        "source /opt/ros/humble/setup.bash",
        f"source {shlex.quote(str(WORKSPACE / 'install/setup.bash'))}",
        shlex.join(args),
    ])
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def resolve_uri(uri):
    if uri.startswith("file://"):
        return uri[len("file://"):]
    if not uri.startswith("package://"):
        return uri

    package_and_path = uri[len("package://"):]
    package, relative = package_and_path.split("/", 1)

    roots = {
        "ur_description": UR_DESCRIPTION,
        "robotiq_ft_sensor_description": FT_DESCRIPTION,
    }

    if package not in roots:
        raise RuntimeError(f"Unconfigured package 未配置包路径: {package}")

    path = roots[package] / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path.resolve())


def strip_ros_tags(root):
    for child in list(root):
        tag = child.tag.split("}")[-1]
        if tag in {"gazebo", "ros2_control", "transmission"}:
            root.remove(child)


def move_base(root):
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            parent is not None
            and child is not None
            and parent.attrib.get("link") == "world"
            and child.attrib.get("link") == "base_link"
        ):
            origin = joint.find("origin")
            if origin is None:
                origin = ET.SubElement(joint, "origin")
            origin.set(
                "xyz",
                f"{ROBOT_BASE_X:.9f} {ROBOT_BASE_Y:.9f} {ROBOT_BASE_Z:.9f}",
            )
            origin.set("rpy", "0 0 0")
            return
    raise RuntimeError("Cannot find world -> base_link joint 找不到机器人基座固定关节")


def add_ft300(root):
    collision_root = FT_DESCRIPTION / "meshes/collision"
    visual_root = FT_DESCRIPTION / "meshes/visual"
    mounting_file = "robotiq_ft300-G-062-COUPLING_G-50-4M6-1D6_20181119.STL"

    ft_collision = collision_root / "robotiq_ft300.STL"
    ft_visual = visual_root / "robotiq_ft300.STL"
    mounting_collision = collision_root / "mountings" / mounting_file
    mounting_visual = visual_root / "mountings" / mounting_file

    for path in [ft_collision, ft_visual, mounting_collision, mounting_visual]:
        if not path.exists():
            raise FileNotFoundError(path)

    mounting = ET.SubElement(root, "link", name="ft300_mounting_plate")
    inertial = ET.SubElement(mounting, "inertial")
    ET.SubElement(
        inertial, "origin",
        xyz="4.7011718298386822e-06 0.00010867051639380253 0.00704862787148917",
        rpy="0 0 0"
    )
    ET.SubElement(inertial, "mass", value="0.043809924362592643")
    ET.SubElement(
        inertial, "inertia",
        ixx="1.3207043280205638e-05", ixy="0", ixz="0",
        iyy="1.3382631571685517e-05", iyz="-1.3899944836228174e-08",
        izz="2.5657101126638417e-05"
    )
    visual = ET.SubElement(mounting, "visual")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(geom, "mesh", filename=str(mounting_visual.resolve()))
    collision = ET.SubElement(mounting, "collision")
    geom = ET.SubElement(collision, "geometry")
    ET.SubElement(geom, "mesh", filename=str(mounting_collision.resolve()))

    joint = ET.SubElement(root, "joint", name="ft300_fix", type="fixed")
    ET.SubElement(joint, "parent", link="tool0")
    ET.SubElement(joint, "child", link="ft300_mounting_plate")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

    sensor = ET.SubElement(root, "link", name="ft300_sensor")
    inertial = ET.SubElement(sensor, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 -0.017", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0.3")
    ET.SubElement(
        inertial, "inertia",
        ixx="0.00026199993681862816", ixy="1.5502376899307433e-12",
        ixz="9.99998186502329e-07", iyy="0.00026500011364938102",
        iyz="1.0000057723466316e-06", izz="0.00021899994953199082"
    )
    visual = ET.SubElement(sensor, "visual")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(geom, "mesh", filename=str(ft_visual.resolve()))
    collision = ET.SubElement(sensor, "collision")
    geom = ET.SubElement(collision, "geometry")
    ET.SubElement(geom, "mesh", filename=str(ft_collision.resolve()))

    joint = ET.SubElement(
        root, "joint",
        name="ft300_mounting_plate_joint",
        type="fixed"
    )
    ET.SubElement(joint, "parent", link="ft300_mounting_plate")
    ET.SubElement(joint, "child", link="ft300_sensor")
    ET.SubElement(
        joint, "origin",
        xyz="0 0 0.0415",
        rpy="0 3.141592653589793 0"
    )

    ET.SubElement(root, "link", name="robotiq_ft_frame_id")
    joint = ET.SubElement(root, "joint", name="measurement_joint", type="fixed")
    ET.SubElement(joint, "parent", link="ft300_sensor")
    ET.SubElement(joint, "child", link="robotiq_ft_frame_id")
    ET.SubElement(
        joint, "origin",
        xyz="0 0 0",
        rpy="0 3.141592653589793 -1.5707963267948966"
    )


def add_probe(root):
    probe = ET.SubElement(root, "link", name="inspection_probe")

    inertial = ET.SubElement(probe, "inertial")
    ET.SubElement(
        inertial, "origin",
        xyz=f"0 0 {PROBE_COM_Z:.17g}",
        rpy="0 0 0"
    )
    ET.SubElement(inertial, "mass", value=f"{PROBE_MASS:.17g}")
    ET.SubElement(
        inertial, "inertia",
        ixx=f"{PROBE_IXX:.17g}", ixy="0", ixz="0",
        iyy=f"{PROBE_IYY:.17g}", iyz="0",
        izz=f"{PROBE_IZZ:.17g}"
    )

    visual = ET.SubElement(probe, "visual", name="inspection_probe_shaft")
    ET.SubElement(
        visual, "origin",
        xyz=f"0 0 {(PROBE_LENGTH - TIP_RADIUS) / 2:.17g}",
        rpy="0 0 0"
    )
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(
        geometry, "cylinder",
        radius=f"{PROBE_RADIUS:.17g}",
        length=f"{PROBE_LENGTH - TIP_RADIUS:.17g}"
    )

    visual = ET.SubElement(probe, "visual", name="inspection_probe_tip_visual")
    ET.SubElement(
        visual, "origin",
        xyz=f"0 0 {PROBE_LENGTH - TIP_RADIUS:.17g}",
        rpy="0 0 0"
    )
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "sphere", radius=f"{TIP_RADIUS:.17g}")

    collision = ET.SubElement(
        probe, "collision",
        name="inspection_probe_tip_collision"
    )
    ET.SubElement(
        collision, "origin",
        xyz=f"0 0 {PROBE_LENGTH - TIP_RADIUS:.17g}",
        rpy="0 0 0"
    )
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "sphere", radius=f"{TIP_RADIUS:.17g}")

    joint = ET.SubElement(root, "joint", name="inspection_probe_joint", type="fixed")
    ET.SubElement(joint, "parent", link="robotiq_ft_frame_id")
    ET.SubElement(joint, "child", link="inspection_probe")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

    ET.SubElement(root, "link", name="inspection_tip")
    joint = ET.SubElement(root, "joint", name="inspection_tip_joint", type="fixed")
    ET.SubElement(joint, "parent", link="inspection_probe")
    ET.SubElement(joint, "child", link="inspection_tip")
    ET.SubElement(
        joint, "origin",
        xyz=f"0 0 {PROBE_LENGTH:.17g}",
        rpy="0 0 0"
    )


def resolve_meshes(root):
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            mesh.set("filename", resolve_uri(filename))


def add_mujoco_extension(root):
    extension = ET.Element("mujoco")
    ET.SubElement(
        extension, "compiler",
        discardvisual="true",
        fusestatic="false",
        balanceinertia="true"
    )
    root.insert(0, extension)


def prepare_resolved_urdf():
    root = ET.fromstring(expand_ur5_xacro())
    root.set("name", "ur5_ft300_inspection")
    strip_ros_tags(root)
    move_base(root)
    add_ft300(root)
    add_probe(root)
    resolve_meshes(root)
    add_mujoco_extension(root)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(
        RESOLVED_URDF,
        encoding="utf-8",
        xml_declaration=True
    )
    return tree


def effort_limits(root):
    result = {}
    for name in ARM_JOINTS:
        joint = root.find(f"joint[@name='{name}']")
        if joint is None:
            raise RuntimeError(f"URDF missing joint URDF缺少关节: {name}")
        limit = joint.find("limit")
        if limit is None or "effort" not in limit.attrib:
            raise RuntimeError(f"URDF missing effort limit: {name}")
        result[name] = float(limit.attrib["effort"])
    return result


def find_body(root, name):
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"MJCF missing body MJCF缺少body: {name}")
    return body


def enhance_mjcf(limits):
    model = mujoco.MjModel.from_xml_path(str(RESOLVED_URDF))
    mujoco.mj_saveLastXML(str(RAW_MJCF), model)

    tree = ET.parse(RAW_MJCF)
    root = tree.getroot()
    root.set("model", "ur5_ft300_inspection")

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

    for name in ARM_JOINTS:
        joint = root.find(f".//joint[@name='{name}']")
        if joint is None:
            raise RuntimeError(f"MJCF missing joint MJCF缺少关节: {name}")
        joint.set("damping", f"{JOINT_DAMPING[name]:g}")
        joint.set("armature", f"{JOINT_ARMATURE[name]:g}")

    for body in root.findall(".//body"):
        body_name = body.attrib.get("name", "")
        for geom in body.findall("geom"):
            geom.set("friction", "0.8 0.01 0.001")

            if body_name == "inspection_probe":
                if geom.attrib.get("name") == "inspection_probe_tip_collision":
                    geom.set("rgba", "0.10 0.12 0.14 1")
                    geom.set("friction", "0.45 0.01 0.001")
                    geom.set("solref", "-2500 -35")
                else:
                    geom.set("rgba", "0.72 0.74 0.77 1")
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
            elif body_name.startswith("ft300"):
                geom.set("rgba", "0.12 0.12 0.14 1")
            elif "wrist" in body_name or "base" in body_name:
                geom.set("rgba", "0.36 0.38 0.40 1")
            else:
                geom.set("rgba", "0.25 0.43 0.58 1")

    # URDF visual geometry is discarded during MuJoCo import, so recreate the
    # probe shaft explicitly as a non-colliding MJCF geom.
    probe_body = find_body(root, "inspection_probe")
    ET.SubElement(
        probe_body,
        "geom",
        name="inspection_probe_shaft",
        type="cylinder",
        size=f"{PROBE_RADIUS:.17g} {(PROBE_LENGTH - TIP_RADIUS) / 2:.17g}",
        pos=f"0 0 {(PROBE_LENGTH - TIP_RADIUS) / 2:.17g}",
        rgba="0.72 0.74 0.77 1",
        contype="0",
        conaffinity="0",
    )

    ft_frame = find_body(root, "robotiq_ft_frame_id")
    ET.SubElement(
        ft_frame, "site",
        name="ft300_site",
        pos="0 0 0",
        size="0.006",
        rgba="1 0.2 0.1 0.8"
    )

    tip_body = find_body(root, "inspection_tip")
    ET.SubElement(
        tip_body, "site",
        name="inspection_tip_site",
        pos="0 0 0",
        size="0.0035",
        rgba="1 0.15 0.05 0.9"
    )

    contact = ET.SubElement(root, "contact")
    bodies = {b.attrib.get("name", "") for b in root.findall(".//body")}
    if "base_link_inertia" in bodies and "shoulder_link" in bodies:
        ET.SubElement(
            contact, "exclude",
            body1="base_link_inertia",
            body2="shoulder_link"
        )

    actuator = ET.SubElement(root, "actuator")
    for actuator_name, joint_name in zip(ACTUATOR_NAMES, ARM_JOINTS):
        limit = limits[joint_name]
        ET.SubElement(
            actuator, "motor",
            name=actuator_name,
            joint=joint_name,
            gear="1",
            ctrllimited="true",
            ctrlrange=f"{-limit:.9g} {limit:.9g}"
        )

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", name="ft300_force", site="ft300_site")
    ET.SubElement(sensor, "torque", name="ft300_torque", site="ft300_site")

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe, "key",
        name="home",
        qpos=" ".join(f"{v:.9g}" for v in HOME_Q),
        ctrl="0 0 0 0 0 0"
    )

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual, "headlight",
        ambient="0.35 0.35 0.35",
        diffuse="0.7 0.7 0.7"
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")

    ET.indent(tree, space="  ")
    tree.write(FINAL_MJCF, encoding="utf-8", xml_declaration=True)


def make_pinocchio_urdf():
    tree = ET.parse(RESOLVED_URDF)
    root = tree.getroot()

    for child in list(root):
        if child.tag == "mujoco":
            root.remove(child)

    model = mujoco.MjModel.from_xml_path(str(FINAL_MJCF))

    for link in root.findall("link"):
        name = link.attrib["name"]
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )
        if body_id < 0 or model.body_mass[body_id] <= 0:
            continue

        old = link.find("inertial")
        if old is not None:
            link.remove(old)

        rot_flat = np.empty(9)
        mujoco.mju_quat2Mat(rot_flat, model.body_iquat[body_id])
        rot = rot_flat.reshape(3, 3)
        inertia = rot @ np.diag(model.body_inertia[body_id]) @ rot.T

        inertial = ET.Element("inertial")
        ET.SubElement(
            inertial, "origin",
            xyz=" ".join(f"{v:.17g}" for v in model.body_ipos[body_id]),
            rpy="0 0 0"
        )
        ET.SubElement(
            inertial, "mass",
            value=f"{model.body_mass[body_id]:.17g}"
        )
        ET.SubElement(
            inertial, "inertia",
            ixx=f"{inertia[0,0]:.17g}",
            ixy=f"{inertia[0,1]:.17g}",
            ixz=f"{inertia[0,2]:.17g}",
            iyy=f"{inertia[1,1]:.17g}",
            iyz=f"{inertia[1,2]:.17g}",
            izz=f"{inertia[2,2]:.17g}"
        )
        link.insert(0, inertial)

    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name not in JOINT_DAMPING:
            continue
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", f"{JOINT_DAMPING[name]:g}")
        dynamics.set("friction", "0")

    root.set("name", "ur5_ft300_inspection")
    ET.indent(tree, space="  ")
    tree.write(PIN_URDF, encoding="utf-8", xml_declaration=True)


def validate_build(limits):
    model = mujoco.MjModel.from_xml_path(str(FINAL_MJCF))
    data = mujoco.MjData(model)

    if (model.nq, model.nv, model.nu, model.nsensor) != (6, 6, 6, 2):
        raise RuntimeError(
            f"Unexpected dimensions: nq={model.nq}, nv={model.nv}, "
            f"nu={model.nu}, nsensor={model.nsensor}"
        )

    home_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    if data.ncon:
        pairs = []
        for c in data.contact:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            pairs.append((g1, g2, float(c.dist)))
        raise RuntimeError(f"Home pose has unexpected contacts: {pairs}")

    tip = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip_site"
    )
    ft = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "ft300_site"
    )

    distance = np.linalg.norm(
        data.site_xpos[tip] - data.site_xpos[ft]
    )

    print("========== 91 UR5 Build 91 UR5检测模型 ==========")
    print("Source robot 机器人源模型: ur_description / ur_type:=ur5")
    print(
        f"Robot base 机器人基座位置(m): "
        f"[{ROBOT_BASE_X:+.3f}, {ROBOT_BASE_Y:+.3f}, {ROBOT_BASE_Z:+.3f}]"
    )
    print(f"MuJoCo model MuJoCo模型: {FINAL_MJCF.name}")
    print(f"Pinocchio model Pinocchio模型: {PIN_URDF.name}")
    print(f"Dimensions 维度: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"FT300-to-TCP distance FT300到TCP距离: {distance*1000:.3f} mm")
    print("URDF effort limits URDF关节力矩上限(Nm):")
    for name in ARM_JOINTS:
        print(f"  {name}: {limits[name]:.3f}")
    print("Home contacts 初始位姿碰撞: 0")
    print("Result 结果: BUILD PASS 生成通过")


def main():
    os.chdir(HERE)
    require_paths()

    tree = prepare_resolved_urdf()
    limits = effort_limits(tree.getroot())

    enhance_mjcf(limits)
    make_pinocchio_urdf()
    validate_build(limits)


if __name__ == "__main__":
    main()
