#!/usr/bin/env python3
"""88 - Build an industrial inspection workcell around the validated robot model.

The robot dynamics are left unchanged. This script only adds static scene
geometry: skybox, floor, workbench, fixture, and a double-curvature ellipsoidal
workpiece. The analytic ellipsoid parameters are also the geometry definition
used by the later raster-scan planner.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent

SOURCE = HERE / "ur3_ft300_inspection.xml"
OUTPUT = HERE / "ur3_ft300_inspection_workcell.xml"

# ---------------------------------------------------------------------------
# Workcell geometry
# ---------------------------------------------------------------------------

TABLE_CENTER = (0.05, -0.25, 0.65)
TABLE_HALF_SIZE = (0.55, 0.50, 0.03)  # table top = 0.68 m

WORKPIECE_CENTER = (0.20, -0.25, 0.765)
WORKPIECE_RADII = (0.16, 0.12, 0.055)
# Bottom = 0.710 m, top = 0.820 m.

FIXTURE_CENTER = (0.20, -0.25, 0.695)
FIXTURE_HALF_SIZE = (0.19, 0.15, 0.015)  # top = 0.710 m

# Later raster scan will stay well inside the ellipsoid boundary.
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
            "Missing ur3_ft300_inspection.xml. "
            "Run 86_build_inspection_probe_model.py first."
        )

    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "ur3_ft300_inspection_workcell")

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    # Non-black background.
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

    add_material(
        asset,
        "floor_mat",
        "0.58 0.61 0.65 1",
        specular="0.10",
        shininess="0.15",
    )
    add_material(
        asset,
        "table_mat",
        "0.22 0.25 0.29 1",
        specular="0.35",
        shininess="0.45",
    )
    add_material(
        asset,
        "fixture_mat",
        "0.12 0.19 0.27 1",
        specular="0.45",
        shininess="0.55",
    )
    add_material(
        asset,
        "workpiece_mat",
        "0.72 0.47 0.18 1",
        specular="0.45",
        shininess="0.50",
    )
    add_material(
        asset,
        "clamp_mat",
        "0.09 0.11 0.14 1",
        specular="0.25",
        shininess="0.30",
    )

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

    # Floor.
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

    # Lighting.
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

    # Workbench body.
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

    # Four legs, from floor to the underside of the top.
    leg_z = (TABLE_CENTER[2] - TABLE_HALF_SIZE[2]) / 2.0
    leg_half_z = leg_z
    leg_half_xy = 0.035
    leg_x = TABLE_CENTER[0] + TABLE_HALF_SIZE[0] - 0.07
    leg_x2 = TABLE_CENTER[0] - TABLE_HALF_SIZE[0] + 0.07
    leg_y = TABLE_CENTER[1] + TABLE_HALF_SIZE[1] - 0.07
    leg_y2 = TABLE_CENTER[1] - TABLE_HALF_SIZE[1] + 0.07

    for index, (x, y) in enumerate(
        [(leg_x, leg_y), (leg_x, leg_y2), (leg_x2, leg_y), (leg_x2, leg_y2)]
    ):
        ET.SubElement(
            bench,
            "geom",
            name=f"table_leg_{index + 1}",
            type="box",
            pos=f"{x:.6f} {y:.6f} {leg_z:.6f}",
            size=f"{leg_half_xy:.6f} {leg_half_xy:.6f} {leg_half_z:.6f}",
            material="table_mat",
            friction="0.9 0.01 0.001",
        )

    # Fixture base under the workpiece.
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

    # Side clamps. They are outside the planned scan region.
    clamp_x_offset = WORKPIECE_RADII[0] + 0.025
    for side, sign in [("left", -1.0), ("right", 1.0)]:
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

    # Double-curvature workpiece.
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

    # Visual markers for the intended scan-area corners. Sites do not collide.
    corners = [
        (-SCAN_X_HALF, -SCAN_Y_HALF),
        (-SCAN_X_HALF, +SCAN_Y_HALF),
        (+SCAN_X_HALF, -SCAN_Y_HALF),
        (+SCAN_X_HALF, +SCAN_Y_HALF),
    ]
    rx, ry, rz = WORKPIECE_RADII
    cx, cy, cz = WORKPIECE_CENTER

    for i, (dx, dy) in enumerate(corners, start=1):
        inside = 1.0 - (dx / rx) ** 2 - (dy / ry) ** 2
        if inside <= 0.0:
            raise RuntimeError("Scan corner lies outside the ellipsoid")
        z = cz + rz * inside ** 0.5
        ET.SubElement(
            bench,
            "site",
            name=f"scan_corner_{i}",
            type="sphere",
            pos=f"{cx + dx:.6f} {cy + dy:.6f} {z + 0.003:.6f}",
            size="0.003",
            rgba="0.10 0.85 0.25 0.95",
        )

    root.insert(
        0,
        ET.Comment(
            " Industrial workcell scene for model-based force-controlled "
            "surface scanning. Robot dynamics are unchanged. "
        ),
    )

    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)

    print("========== 88 Workcell Build 88工业检测场景 ==========")
    print(f"Scene 场景: {OUTPUT.name}")
    print("Background 背景: gradient skybox 渐变工业背景")
    print("Workbench 工作台: added 已添加")
    print("Fixture 夹具: added 已添加")
    print("Workpiece 工件: double-curvature ellipsoid 双曲率椭球曲面")
    print(
        "Workpiece center 工件中心(m): "
        f"[{cx:+.3f}, {cy:+.3f}, {cz:+.3f}]"
    )
    print(
        "Workpiece radii 工件半轴(m): "
        f"[{rx:.3f}, {ry:.3f}, {rz:.3f}]"
    )
    print(
        "Planned scan area 计划扫描区域: "
        f"{2 * SCAN_X_HALF * 1000:.0f} x "
        f"{2 * SCAN_Y_HALF * 1000:.0f} mm"
    )


if __name__ == "__main__":
    main()
