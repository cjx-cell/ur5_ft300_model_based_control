#!/usr/bin/env python3
"""89 - Generate a 3D raster scan path and desired tool orientation.

The path lies on the upper half of the ellipsoidal workpiece created in step 88.

Conventions
-----------
- Surface normal n points outward from the workpiece.
- The inspection probe local +Z axis points from FT300 toward the probe tip.
- During contact, desired tool +Z therefore points INTO the workpiece:

      z_tool_des = -n

- Tool +X is chosen from the positive ellipsoid x-parameter direction rather
  than the instantaneous travel direction. This avoids 180-degree yaw flips
  when the raster scan reverses direction on alternate rows.
"""

from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent

SOURCE_SCENE = HERE / "ur3_ft300_inspection_workcell.xml"

OUTPUT_NPZ = HERE / "89_raster_scan_path.npz"
OUTPUT_CSV = HERE / "89_raster_scan_path.csv"
OUTPUT_PREVIEW = HERE / "ur3_ft300_inspection_scan_preview.xml"

# Must match step 88.
CENTER = np.array([0.20, -0.25, 0.765], dtype=float)
RADII = np.array([0.16, 0.12, 0.055], dtype=float)

X_HALF = 0.10
Y_HALF = 0.06

ROW_SPACING = 0.020
POINTS_PER_ROW = 121
POINTS_PER_CONNECTOR = 21

PATH_MARKER_STRIDE = 5
NORMAL_MARKER_STRIDE = 40
NORMAL_DISPLAY_LENGTH = 0.025


def surface_height(x: float, y: float) -> float:
    cx, cy, cz = CENTER
    rx, ry, rz = RADII

    dx = x - cx
    dy = y - cy

    inside = 1.0 - (dx / rx) ** 2 - (dy / ry) ** 2
    if inside <= 0.0:
        raise ValueError(
            f"Point outside ellipsoid projection 曲面点越界: x={x}, y={y}"
        )

    return cz + rz * np.sqrt(inside)


def surface_normal(x: float, y: float, z: float) -> np.ndarray:
    """Outward normal of the ellipsoid."""
    c = CENTER
    r = RADII
    p = np.array([x, y, z], dtype=float)

    n = (p - c) / (r * r)
    return n / np.linalg.norm(n)


def positive_x_tangent(x: float, y: float) -> np.ndarray:
    """Tangent for increasing ellipsoid x parameter on the upper surface."""
    cx, cy, _ = CENTER
    rx, ry, rz = RADII

    dx = x - cx
    dy = y - cy
    inside = 1.0 - (dx / rx) ** 2 - (dy / ry) ** 2

    dz_dx = -rz * dx / (rx * rx * np.sqrt(inside))

    t = np.array([1.0, 0.0, dz_dx], dtype=float)
    return t / np.linalg.norm(t)


def desired_rotation(x: float, y: float, z: float) -> np.ndarray:
    """World rotation matrix [x_tool, y_tool, z_tool]."""
    n = surface_normal(x, y, z)

    z_tool = -n

    x_seed = positive_x_tangent(x, y)

    # Numerical projection makes the definition robust even if the analytic
    # expressions are modified later.
    x_tool = x_seed - np.dot(x_seed, z_tool) * z_tool
    x_tool /= np.linalg.norm(x_tool)

    y_tool = np.cross(z_tool, x_tool)
    y_tool /= np.linalg.norm(y_tool)

    # Re-orthogonalize x to remove accumulated floating-point error.
    x_tool = np.cross(y_tool, z_tool)
    x_tool /= np.linalg.norm(x_tool)

    return np.column_stack([x_tool, y_tool, z_tool])


def append_point(points, row_ids, phases, x, y, row_id, phase):
    z = surface_height(x, y)
    p = np.array([x, y, z], dtype=float)

    if points and np.linalg.norm(p - points[-1]) < 1e-12:
        return

    points.append(p)
    row_ids.append(row_id)
    phases.append(phase)


def build_path():
    y_rows = np.arange(
        CENTER[1] - Y_HALF,
        CENTER[1] + Y_HALF + 0.5 * ROW_SPACING,
        ROW_SPACING,
    )

    points = []
    row_ids = []
    phases = []

    for row_index, y in enumerate(y_rows):
        if row_index % 2 == 0:
            x_values = np.linspace(
                CENTER[0] - X_HALF,
                CENTER[0] + X_HALF,
                POINTS_PER_ROW,
            )
        else:
            x_values = np.linspace(
                CENTER[0] + X_HALF,
                CENTER[0] - X_HALF,
                POINTS_PER_ROW,
            )

        for x in x_values:
            append_point(
                points,
                row_ids,
                phases,
                float(x),
                float(y),
                row_index,
                0,  # scan row
            )

        if row_index < len(y_rows) - 1:
            x_end = float(x_values[-1])
            y_next = float(y_rows[row_index + 1])

            connector_y = np.linspace(
                float(y),
                y_next,
                POINTS_PER_CONNECTOR,
            )[1:]

            for yc in connector_y:
                append_point(
                    points,
                    row_ids,
                    phases,
                    x_end,
                    float(yc),
                    row_index,
                    1,  # row connector
                )

    return (
        np.asarray(points),
        np.asarray(row_ids, dtype=int),
        np.asarray(phases, dtype=int),
        y_rows,
    )


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def path_tangent(points: np.ndarray, s: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(points)

    for axis in range(3):
        tangent[:, axis] = np.gradient(points[:, axis], s)

    norms = np.linalg.norm(tangent, axis=1)
    tangent /= norms[:, None]
    return tangent


def write_csv(
    points,
    normals,
    motion_tangents,
    rotations,
    s,
    row_ids,
    phases,
):
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "s_m",
                "x_m",
                "y_m",
                "z_m",
                "normal_x",
                "normal_y",
                "normal_z",
                "motion_tangent_x",
                "motion_tangent_y",
                "motion_tangent_z",
                "tool_z_x",
                "tool_z_y",
                "tool_z_z",
                "row_id",
                "phase",
            ]
        )

        for i in range(len(points)):
            writer.writerow(
                [
                    i,
                    s[i],
                    *points[i],
                    *normals[i],
                    *motion_tangents[i],
                    *rotations[i, :, 2],
                    int(row_ids[i]),
                    int(phases[i]),
                ]
            )


def add_preview_geometry(points, normals):
    tree = ET.parse(SOURCE_SCENE)
    root = tree.getroot()
    root.set("model", "ur3_ft300_inspection_scan_preview")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no worldbody")

    preview = ET.SubElement(worldbody, "body", name="scan_path_preview")

    # Draw the path with thin non-colliding capsules.
    indices = list(range(0, len(points), PATH_MARKER_STRIDE))
    if indices[-1] != len(points) - 1:
        indices.append(len(points) - 1)

    for k in range(len(indices) - 1):
        i = indices[k]
        j = indices[k + 1]
        p0 = points[i] + 0.0015 * normals[i]
        p1 = points[j] + 0.0015 * normals[j]

        ET.SubElement(
            preview,
            "geom",
            name=f"scan_path_segment_{k:04d}",
            type="capsule",
            fromto=(
                f"{p0[0]:.8f} {p0[1]:.8f} {p0[2]:.8f} "
                f"{p1[0]:.8f} {p1[1]:.8f} {p1[2]:.8f}"
            ),
            size="0.0012",
            rgba="0.05 0.85 0.25 0.95",
            contype="0",
            conaffinity="0",
        )

    # Draw sparse outward surface normals.
    for marker_id, i in enumerate(
        range(0, len(points), NORMAL_MARKER_STRIDE)
    ):
        p0 = points[i] + 0.0020 * normals[i]
        p1 = p0 + NORMAL_DISPLAY_LENGTH * normals[i]

        ET.SubElement(
            preview,
            "geom",
            name=f"surface_normal_{marker_id:03d}",
            type="capsule",
            fromto=(
                f"{p0[0]:.8f} {p0[1]:.8f} {p0[2]:.8f} "
                f"{p1[0]:.8f} {p1[1]:.8f} {p1[2]:.8f}"
            ),
            size="0.0010",
            rgba="0.10 0.35 1.00 0.90",
            contype="0",
            conaffinity="0",
        )

    # Start / end markers.
    ET.SubElement(
        preview,
        "site",
        name="scan_start",
        type="sphere",
        pos=" ".join(f"{v:.8f}" for v in points[0]),
        size="0.005",
        rgba="0.05 1.0 0.05 1",
    )
    ET.SubElement(
        preview,
        "site",
        name="scan_end",
        type="sphere",
        pos=" ".join(f"{v:.8f}" for v in points[-1]),
        size="0.005",
        rgba="1.0 0.10 0.10 1",
    )

    root.insert(
        0,
        ET.Comment(
            " Green path: raster scan. Blue markers: outward surface normals. "
        ),
    )

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_PREVIEW, encoding="utf-8", xml_declaration=True)


def main():
    if not SOURCE_SCENE.exists():
        raise FileNotFoundError(
            "Missing ur3_ft300_inspection_workcell.xml. "
            "Run step 88 first."
        )

    points, row_ids, phases, y_rows = build_path()
    s = cumulative_arc_length(points)
    motion_tangents = path_tangent(points, s)

    normals = np.array(
        [surface_normal(*p) for p in points],
        dtype=float,
    )
    rotations = np.array(
        [desired_rotation(*p) for p in points],
        dtype=float,
    )

    tool_z = rotations[:, :, 2]

    normal_alignment = np.sum(tool_z * normals, axis=1)

    orthonormal_error = max(
        np.linalg.norm(R.T @ R - np.eye(3), ord=np.inf)
        for R in rotations
    )

    determinant_error = float(
        np.max(np.abs(np.linalg.det(rotations) - 1.0))
    )

    max_tool_normal_error_deg = float(
        np.degrees(
            np.max(
                np.arccos(
                    np.clip(-normal_alignment, -1.0, 1.0)
                )
            )
        )
    )

    np.savez(
        OUTPUT_NPZ,
        position=points,
        surface_normal=normals,
        motion_tangent=motion_tangents,
        rotation=rotations,
        tool_z=tool_z,
        arc_length=s,
        row_id=row_ids,
        phase=phases,
        workpiece_center=CENTER,
        workpiece_radii=RADII,
        x_half=X_HALF,
        y_half=Y_HALF,
        row_spacing=ROW_SPACING,
    )

    write_csv(
        points,
        normals,
        motion_tangents,
        rotations,
        s,
        row_ids,
        phases,
    )
    add_preview_geometry(points, normals)

    print("========== 89 Raster Path 89三维往复式扫描轨迹 ==========")
    print(f"Rows 扫描行数: {len(y_rows)}")
    print(f"Path points 路径点数: {len(points)}")
    print(f"Path length 路径总长度: {s[-1] * 1000.0:.1f} mm")
    print(
        f"Scan area 扫描区域: "
        f"{2 * X_HALF * 1000.0:.0f} x "
        f"{2 * Y_HALF * 1000.0:.0f} mm"
    )
    print(
        f"Row spacing 行间距: {ROW_SPACING * 1000.0:.1f} mm"
    )
    print(
        f"Tool +Z vs -normal 工具+Z与负法向最大误差: "
        f"{max_tool_normal_error_deg:.3e} deg"
    )
    print(
        f"Rotation orthonormal error 旋转矩阵正交误差: "
        f"{orthonormal_error:.3e}"
    )
    print(
        f"Rotation determinant error 旋转矩阵行列式误差: "
        f"{determinant_error:.3e}"
    )
    print(f"Saved 保存: {OUTPUT_NPZ.name}")
    print(f"Saved 保存: {OUTPUT_CSV.name}")
    print(f"Preview 预览模型: {OUTPUT_PREVIEW.name}")


if __name__ == "__main__":
    main()
