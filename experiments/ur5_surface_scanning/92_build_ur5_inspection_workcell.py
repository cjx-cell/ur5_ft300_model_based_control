#!/usr/bin/env python3
"""92 - Build the UR5 industrial surface-scanning workcell.

This reuses exactly the same workpiece geometry and scan-area definition that
were used for the UR3 comparison.  Only the robot model is changed to UR5.

Robot source:
    ur5_ft300_inspection.xml

Output:
    ur5_ft300_inspection_workcell.xml
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


HERE = Path(__file__).resolve().parent

SOURCE = HERE / "ur5_ft300_inspection.xml"
OUTPUT = HERE / "ur5_ft300_inspection_workcell.xml"

# ---------------------------------------------------------------------------
# Keep the workcell and workpiece identical to the UR3 experiment.
# ---------------------------------------------------------------------------

TABLE_CENTER = (0.05, -0.25, 0.65)
TABLE_HALF_SIZE = (0.55, 0.50, 0.03)  # top = 0.68 m

# UR5 base frame was placed at z = 0.76 m in step 91.
PEDESTAL_CENTER = (-0.20, 0.05, 0.72)
PEDESTAL_HALF_SIZE = (0.105, 0.105, 0.04)

FIXTURE_CENTER = (0.20, -0.25, 0.695)
FIXTURE_HALF_SIZE = (0.19, 0.15, 0.015)

WORKPIECE_CENTER = (0.20, -0.25, 0.765)
WORKPIECE_RADII = (0.16, 0.12, 0.055)

SCAN_X_HALF = 0.10
SCAN_Y_HALF = 0.06


def add_material(asset, name, rgba, specular="0.25", shininess="0.35"):
    ET.SubElement(
        asset,
        "material",
        name=name,
        rgba=rgba,
        specular=specular,
        shininess=shininess,
    )


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            "Missing ur5_ft300_inspection.xml. "
            "Run 91_build_ur5_ft300_inspection_model.py first."
        )

    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "ur5_ft300_inspection_workcell")

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    ET.SubElement(
        asset,
        "texture",
        name="workcell_skybox",
        type="skybox",
        builtin="gradient",
        rgb1="0.92 0.95 1.00",
        rgb2="0.45 0.56 0.72",
        width="512",
        height="3072",
    )

    add_material(asset, "floor_mat", "0.58 0.61 0.65 1", "0.10", "0.15")
    add_material(asset, "table_mat", "0.22 0.25 0.29 1", "0.35", "0.45")
    add_material(asset, "pedestal_mat", "0.30 0.32 0.35 1", "0.35", "0.40")
    add_material(asset, "fixture_mat", "0.12 0.19 0.27 1", "0.45", "0.55")
    add_material(asset, "workpiece_mat", "0.72 0.47 0.18 1", "0.45", "0.50")
    add_material(asset, "clamp_mat", "0.09 0.11 0.14 1", "0.25", "0.30")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    headlight = visual.find("headlight")
    if headlight is not None:
        headlight.set("ambient", "0.45 0.45 0.45")
        headlight.set("diffuse", "0.75 0.75 0.75")

    rgba = visual.find("rgba")
    if rgba is not None:
        rgba.set("haze", "0.75 0.82 0.90 1")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no worldbody")

    # ------------------------------------------------------------------
    # Factory floor
    # ------------------------------------------------------------------
    ET.SubElement(
        worldbody,
        "geom",
        name="factory_floor",
        type="plane",
        size="3 3 0.1",
        pos="0 0 0",
        material="floor_mat",
        friction="0.9 0.01 0.001",
    )

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------
    ET.SubElement(
        worldbody,
        "light",
        name="key_light",
        pos="0.30 -0.30 2.40",
        dir="-0.10 0.05 -1",
        directional="true",
        diffuse="0.90 0.90 0.90",
        specular="0.25 0.25 0.25",
    )

    ET.SubElement(
        worldbody,
        "light",
        name="fill_light",
        pos="-1.20 0.40 1.60",
        dir="0.60 -0.20 -0.70",
        directional="true",
        diffuse="0.45 0.48 0.55",
        specular="0.10 0.10 0.12",
    )

    # ------------------------------------------------------------------
    # Workbench
    # ------------------------------------------------------------------
    bench = ET.SubElement(worldbody, "body", name="inspection_workbench")

    ET.SubElement(
        bench,
        "geom",
        name="table_top",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in TABLE_CENTER),
        size=" ".join(f"{v:.6f}" for v in TABLE_HALF_SIZE),
        material="table_mat",
        friction="0.9 0.01 0.001",
    )

    leg_z = (TABLE_CENTER[2] - TABLE_HALF_SIZE[2]) / 2.0
    leg_half_z = leg_z
    leg_half_xy = 0.035

    x_pos = TABLE_CENTER[0] + TABLE_HALF_SIZE[0] - 0.07
    x_neg = TABLE_CENTER[0] - TABLE_HALF_SIZE[0] + 0.07
    y_pos = TABLE_CENTER[1] + TABLE_HALF_SIZE[1] - 0.07
    y_neg = TABLE_CENTER[1] - TABLE_HALF_SIZE[1] + 0.07

    for index, (x, y) in enumerate(
        [(x_pos, y_pos), (x_pos, y_neg), (x_neg, y_pos), (x_neg, y_neg)],
        start=1,
    ):
        ET.SubElement(
            bench,
            "geom",
            name=f"table_leg_{index}",
            type="box",
            pos=f"{x:.6f} {y:.6f} {leg_z:.6f}",
            size=f"{leg_half_xy:.6f} {leg_half_xy:.6f} {leg_half_z:.6f}",
            material="table_mat",
            friction="0.9 0.01 0.001",
        )

    # Visual mounting riser from table top to the robot base frame.
    # It is non-colliding so it does not create an artificial base contact.
    ET.SubElement(
        bench,
        "geom",
        name="robot_pedestal",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in PEDESTAL_CENTER),
        size=" ".join(f"{v:.6f}" for v in PEDESTAL_HALF_SIZE),
        material="pedestal_mat",
        contype="0",
        conaffinity="0",
    )

    # ------------------------------------------------------------------
    # Fixture
    # ------------------------------------------------------------------
    ET.SubElement(
        bench,
        "geom",
        name="fixture_base",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in FIXTURE_CENTER),
        size=" ".join(f"{v:.6f}" for v in FIXTURE_HALF_SIZE),
        material="fixture_mat",
        friction="0.8 0.01 0.001",
    )

    clamp_x_offset = WORKPIECE_RADII[0] + 0.025

    for side, sign in [("left", -1.0), ("right", +1.0)]:
        ET.SubElement(
            bench,
            "geom",
            name=f"fixture_clamp_{side}",
            type="box",
            pos=(
                f"{WORKPIECE_CENTER[0] + sign * clamp_x_offset:.6f} "
                f"{WORKPIECE_CENTER[1]:.6f} "
                f"{0.725:.6f}"
            ),
            size="0.018 0.135 0.030",
            material="clamp_mat",
            friction="0.8 0.01 0.001",
        )

    # ------------------------------------------------------------------
    # Exact same double-curvature workpiece as UR3 test.
    # ------------------------------------------------------------------
    ET.SubElement(
        bench,
        "geom",
        name="inspection_workpiece",
        type="ellipsoid",
        pos=" ".join(f"{v:.6f}" for v in WORKPIECE_CENTER),
        size=" ".join(f"{v:.6f}" for v in WORKPIECE_RADII),
        material="workpiece_mat",
        friction="0.30 0.01 0.001",
        solref="-2500 -35",
    )

    # Scan-area corner markers.
    cx, cy, cz = WORKPIECE_CENTER
    rx, ry, rz = WORKPIECE_RADII

    corners = [
        (-SCAN_X_HALF, -SCAN_Y_HALF),
        (-SCAN_X_HALF, +SCAN_Y_HALF),
        (+SCAN_X_HALF, -SCAN_Y_HALF),
        (+SCAN_X_HALF, +SCAN_Y_HALF),
    ]

    for index, (dx, dy) in enumerate(corners, start=1):
        inside = 1.0 - (dx / rx) ** 2 - (dy / ry) ** 2
        if inside <= 0.0:
            raise RuntimeError("Scan corner lies outside ellipsoid")

        z = cz + rz * np.sqrt(inside)

        ET.SubElement(
            bench,
            "site",
            name=f"scan_corner_{index}",
            type="sphere",
            pos=f"{cx + dx:.6f} {cy + dy:.6f} {z + 0.003:.6f}",
            size="0.003",
            rgba="0.10 0.85 0.25 0.95",
        )

    root.insert(
        0,
        ET.Comment(
            " UR5 workcell. Workpiece geometry and scan area are unchanged "
            "from the UR3 reachability experiment for a controlled comparison. "
        ),
    )

    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)

    # Reload once so XML/schema problems are caught immediately.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    tip_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip_site"
    )
    workpiece_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "inspection_workpiece"
    )

    tip_position = data.site_xpos[tip_site_id].copy()
    workpiece_center = data.geom_xpos[workpiece_id].copy()
    workpiece_top = workpiece_center[2] + model.geom_size[workpiece_id, 2]

    print("========== 92 UR5 Workcell 92 UR5工业扫描场景 ==========")
    print(f"Scene 场景: {OUTPUT.name}")
    print("Robot 机器人: UR5 + FT300 + inspection probe")
    print("Workpiece 工件: same position/path as before 工件与扫描路径位置保持不变")
    base_xy = np.array([-0.20, 0.05])
    workpiece_xy = np.array(WORKPIECE_CENTER[:2])
    print(
        f"Base-to-workpiece horizontal distance 基座到工件水平距离: "
        f"{np.linalg.norm(workpiece_xy - base_xy) * 1000.0:.1f} mm"
    )
    print(
        f"Scan area 扫描区域: "
        f"{2 * SCAN_X_HALF * 1000:.0f} x "
        f"{2 * SCAN_Y_HALF * 1000:.0f} mm"
    )
    print(
        f"Home TCP Z 初始TCP高度: {tip_position[2]:.3f} m"
    )
    print(
        f"Home TCP clearance 初始TCP距工件顶部: "
        f"{(tip_position[2] - workpiece_top) * 1000:.1f} mm"
    )
    print(f"Home contacts 初始碰撞数: {data.ncon}")
    print("Result 结果: BUILD PASS 生成通过")


if __name__ == "__main__":
    main()
