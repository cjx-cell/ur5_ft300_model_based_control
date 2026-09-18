#!/usr/bin/env python3
"""97B - Robust 3D surface reconstruction from 97A tactile exploration.

This script reconstructs a smooth surface model using ONLY the tactile data
collected by 97A.

Training input
--------------
97_unknown_surface_points.npz

Reconstruction input from step 89
---------------------------------
Only the XY raster layout / row_id / phase are used as query locations.

The following step-89 fields are NOT used for fitting:
- true Z
- true surface normal
- true rotation

They are loaded only AFTER reconstruction to report evaluation metrics.

Method
------
For every raster XY query point, fit a local quadratic surface

    z = a0 + a1*u + a2*v + a3*u^2 + a4*u*v + a5*v^2

where u = (x-xq)/S and v = (y-yq)/S.

Weights combine:
- spatial Gaussian weighting,
- FT300 force-quality weighting,
- TCP tracking-quality weighting,
- reduced weight for connector-phase samples,
- robust IRLS residual weighting.

Outputs
-------
97B_reconstructed_surface.npz
97B_reconstructed_surface.csv
"""

from pathlib import Path
import csv

import numpy as np


HERE = Path(__file__).resolve().parent

EXPLORATION_PATH = HERE / "97_unknown_surface_points.npz"
RASTER_PATH = HERE / "89_raster_scan_path.npz"

NPZ_OUTPUT = HERE / "97B_reconstructed_surface.npz"
CSV_OUTPUT = HERE / "97B_reconstructed_surface.csv"


# ---------------------------------------------------------------------------
# Local robust fit settings
# ---------------------------------------------------------------------------

NEIGHBOR_COUNT = 120
LOCAL_SCALE = 0.025          # m
GAUSSIAN_BANDWIDTH = 0.040   # m

FORCE_ERROR_SCALE = 0.60     # N
TCP_ERROR_SCALE = 0.00050    # m

CONNECTOR_WEIGHT = 0.35

IRLS_ITERATIONS = 5
HUBER_K = 1.5

MIN_WEIGHT = 1e-6


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))

    if n < 1e-12:
        raise RuntimeError("零向量无法归一化")

    return v / n


def safe_angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)

    cosine = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def huber_weights(residual, scale):
    residual = np.asarray(
        residual,
        dtype=float,
    )

    scale = max(
        float(scale),
        1e-9,
    )

    threshold = (
        HUBER_K
        * scale
    )

    absolute = np.abs(
        residual
    )

    weights = np.ones_like(
        absolute
    )

    mask = (
        absolute
        > threshold
    )

    weights[
        mask
    ] = (
        threshold
        / absolute[
            mask
        ]
    )

    return weights


def weighted_lstsq(
    A,
    z,
    weights,
):
    weights = np.maximum(
        np.asarray(
            weights,
            dtype=float,
        ),
        MIN_WEIGHT,
    )

    sqrt_w = np.sqrt(
        weights
    )

    Aw = (
        A
        * sqrt_w[
            :,
            None
        ]
    )

    zw = (
        z
        * sqrt_w
    )

    coefficients, *_ = np.linalg.lstsq(
        Aw,
        zw,
        rcond=None,
    )

    return coefficients


def local_quadratic_fit(
    query_xy,
    sample_xy,
    sample_z,
    base_quality,
):
    delta = (
        sample_xy
        - query_xy[
            None,
            :
        ]
    )

    distance = np.linalg.norm(
        delta,
        axis=1,
    )

    count = min(
        NEIGHBOR_COUNT,
        len(
            sample_xy
        ),
    )

    neighbor_index = np.argpartition(
        distance,
        count - 1,
    )[
        :count
    ]

    local_delta = (
        delta[
            neighbor_index
        ]
    )

    local_distance = (
        distance[
            neighbor_index
        ]
    )

    local_z = (
        sample_z[
            neighbor_index
        ]
    )

    u = (
        local_delta[
            :,
            0
        ]
        / LOCAL_SCALE
    )

    v = (
        local_delta[
            :,
            1
        ]
        / LOCAL_SCALE
    )

    A = np.column_stack(
        [
            np.ones_like(
                u
            ),
            u,
            v,
            u**2,
            u * v,
            v**2,
        ]
    )

    spatial_weight = np.exp(
        -0.5
        * (
            local_distance
            / GAUSSIAN_BANDWIDTH
        )**2
    )

    quality_weight = (
        base_quality[
            neighbor_index
        ]
    )

    weights = (
        spatial_weight
        * quality_weight
    )

    coefficients = weighted_lstsq(
        A,
        local_z,
        weights,
    )

    for _ in range(
        IRLS_ITERATIONS
    ):
        residual = (
            local_z
            - A
            @ coefficients
        )

        median = np.median(
            residual
        )

        mad = np.median(
            np.abs(
                residual
                - median
            )
        )

        robust_scale = max(
            1.4826
            * mad,
            2e-5,
        )

        robust_weight = huber_weights(
            residual,
            robust_scale,
        )

        weights = (
            spatial_weight
            * quality_weight
            * robust_weight
        )

        coefficients = weighted_lstsq(
            A,
            local_z,
            weights,
        )

    residual = (
        local_z
        - A
        @ coefficients
    )

    weighted_rms = float(
        np.sqrt(
            np.sum(
                weights
                * residual**2
            )
            / np.sum(
                weights
            )
        )
    )

    # Because the local coordinates are centered at the query:
    #
    # z(query) = a0
    #
    # dz/dx = a1 / LOCAL_SCALE
    # dz/dy = a2 / LOCAL_SCALE
    reconstructed_z = float(
        coefficients[
            0
        ]
    )

    dz_dx = float(
        coefficients[
            1
        ]
        / LOCAL_SCALE
    )

    dz_dy = float(
        coefficients[
            2
        ]
        / LOCAL_SCALE
    )

    normal = normalize(
        np.array(
            [
                -dz_dx,
                -dz_dy,
                1.0,
            ],
            dtype=float,
        )
    )

    if normal[
        2
    ] < 0.0:
        normal *= -1.0

    effective_weight = float(
        np.sum(
            weights
        )
    )

    return (
        reconstructed_z,
        normal,
        coefficients,
        weighted_rms,
        effective_weight,
    )


def orientation_from_normal(
    normal,
):
    normal = normalize(
        normal
    )

    # Probe +Z points inward.
    z_tool = (
        -normal
    )

    # Keep a globally consistent roll by projecting world +X
    # onto the tangent plane.
    x_reference = np.array(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    x_tool = (
        x_reference
        - np.dot(
            x_reference,
            z_tool,
        )
        * z_tool
    )

    if np.linalg.norm(
        x_tool
    ) < 1e-8:
        x_reference = np.array(
            [
                0.0,
                1.0,
                0.0,
            ]
        )

        x_tool = (
            x_reference
            - np.dot(
                x_reference,
                z_tool,
            )
            * z_tool
        )

    x_tool = normalize(
        x_tool
    )

    y_tool = normalize(
        np.cross(
            z_tool,
            x_tool,
        )
    )

    # Re-orthogonalize x.
    x_tool = normalize(
        np.cross(
            y_tool,
            z_tool,
        )
    )

    R = np.column_stack(
        [
            x_tool,
            y_tool,
            z_tool,
        ]
    )

    return R


def compute_arc_length(
    positions,
):
    step = np.linalg.norm(
        np.diff(
            positions,
            axis=0,
        ),
        axis=1,
    )

    return np.concatenate(
        [
            np.zeros(
                1
            ),
            np.cumsum(
                step
            ),
        ]
    )


def main():
    for path in [
        EXPLORATION_PATH,
        RASTER_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    exploration = np.load(
        EXPLORATION_PATH
    )

    raster = np.load(
        RASTER_PATH
    )

    required_exploration = {
        "position",
        "force_vertical",
        "tcp_tracking_error",
        "phase",
    }

    missing_exploration = (
        required_exploration
        - set(
            exploration.files
        )
    )

    if missing_exploration:
        raise RuntimeError(
            "97A数据缺少字段: "
            f"{sorted(missing_exploration)}; "
            f"actual={exploration.files}"
        )

    required_raster = {
        "position",
        "row_id",
        "phase",
    }

    missing_raster = (
        required_raster
        - set(
            raster.files
        )
    )

    if missing_raster:
        raise RuntimeError(
            "89路径数据缺少字段: "
            f"{sorted(missing_raster)}; "
            f"actual={raster.files}"
        )

    measured_position = np.asarray(
        exploration[
            "position"
        ],
        dtype=float,
    )

    measured_xy = measured_position[
        :,
        :2
    ]

    measured_z = measured_position[
        :,
        2
    ]

    measured_force = np.asarray(
        exploration[
            "force_vertical"
        ],
        dtype=float,
    )

    tcp_error = np.asarray(
        exploration[
            "tcp_tracking_error"
        ],
        dtype=float,
    )

    measured_phase = np.asarray(
        exploration[
            "phase"
        ],
        dtype=int,
    )

    desired_force = (
        float(
            exploration[
                "desired_force"
            ][0]
        )
        if "desired_force"
        in exploration.files
        else 1.0
    )

    query_xy = np.asarray(
        raster[
            "position"
        ][
            :,
            :2
        ],
        dtype=float,
    )

    row_id = np.asarray(
        raster[
            "row_id"
        ],
        dtype=int,
    )

    phase = np.asarray(
        raster[
            "phase"
        ],
        dtype=int,
    )

    if (
        len(
            measured_xy
        )
        != len(
            query_xy
        )
    ):
        raise RuntimeError(
            "97A点数与89路径点数不一致: "
            f"{len(measured_xy)} vs "
            f"{len(query_xy)}"
        )

    # -----------------------------------------------------------------------
    # Base measurement-quality weights.
    # No ground-truth surface information is used here.
    # -----------------------------------------------------------------------
    force_error = np.abs(
        measured_force
        - desired_force
    )

    force_quality = (
        1.0
        / (
            1.0
            + (
                force_error
                / FORCE_ERROR_SCALE
            )**2
        )
    )

    tcp_quality = (
        1.0
        / (
            1.0
            + (
                tcp_error
                / TCP_ERROR_SCALE
            )**2
        )
    )

    phase_quality = np.where(
        measured_phase
        == 0,
        1.0,
        CONNECTOR_WEIGHT,
    )

    contact_quality = np.where(
        measured_force
        >= 0.15,
        1.0,
        0.08,
    )

    base_quality = (
        force_quality
        * tcp_quality
        * phase_quality
        * contact_quality
    )

    reconstructed_z = np.empty(
        len(
            query_xy
        )
    )

    reconstructed_normal = np.empty(
        (
            len(
                query_xy
            ),
            3,
        )
    )

    coefficients = np.empty(
        (
            len(
                query_xy
            ),
            6,
        )
    )

    fit_rms = np.empty(
        len(
            query_xy
        )
    )

    fit_weight = np.empty(
        len(
            query_xy
        )
    )

    print(
        "========== 97B Surface Reconstruction "
        "97B未知曲面重建 =========="
    )

    print(
        f"Input tactile points 输入触觉点数: "
        f"{len(measured_xy)}"
    )

    print(
        f"Desired exploration force "
        f"97A目标探测力: "
        f"{desired_force:.2f} N"
    )

    print(
        f"Local fit neighbors "
        f"局部拟合邻点数: "
        f"{NEIGHBOR_COUNT}"
    )

    print(
        "Training truth usage 训练时真实曲面使用: "
        "NONE 无"
    )

    for index in range(
        len(
            query_xy
        )
    ):
        (
            reconstructed_z[
                index
            ],
            reconstructed_normal[
                index
            ],
            coefficients[
                index
            ],
            fit_rms[
                index
            ],
            fit_weight[
                index
            ],
        ) = local_quadratic_fit(
            query_xy[
                index
            ],
            measured_xy,
            measured_z,
            base_quality,
        )

        if (
            index == 0
            or (
                index + 1
            )
            % 100
            == 0
            or index
            == len(
                query_xy
            )
            - 1
        ):
            print(
                f"Progress 进度: "
                f"{index + 1:4d}/{len(query_xy)} | "
                f"z={reconstructed_z[index]:.6f} m | "
                f"fit_rms={fit_rms[index] * 1000.0:.3f} mm | "
                f"n=["
                f"{reconstructed_normal[index,0]:+.3f}, "
                f"{reconstructed_normal[index,1]:+.3f}, "
                f"{reconstructed_normal[index,2]:+.3f}]"
            )

    reconstructed_position = np.column_stack(
        [
            query_xy,
            reconstructed_z,
        ]
    )

    rotations = np.empty(
        (
            len(
                query_xy
            ),
            3,
            3,
        )
    )

    tool_z = np.empty(
        (
            len(
                query_xy
            ),
            3,
        )
    )

    for index in range(
        len(
            query_xy
        )
    ):
        rotations[
            index
        ] = orientation_from_normal(
            reconstructed_normal[
                index
            ]
        )

        tool_z[
            index
        ] = rotations[
            index,
            :,
            2,
        ]

    arc_length = compute_arc_length(
        reconstructed_position
    )

    tool_alignment_error = []

    orthonormal_error = []
    determinant_error = []

    for index in range(
        len(
            query_xy
        )
    ):
        tool_alignment_error.append(
            safe_angle_deg(
                tool_z[
                    index
                ],
                -reconstructed_normal[
                    index
                ],
            )
        )

        R = rotations[
            index
        ]

        orthonormal_error.append(
            float(
                np.linalg.norm(
                    R.T
                    @ R
                    - np.eye(
                        3
                    )
                )
            )
        )

        determinant_error.append(
            abs(
                float(
                    np.linalg.det(
                        R
                    )
                )
                - 1.0
            )
        )

    # -----------------------------------------------------------------------
    # Evaluation ONLY.
    #
    # The true step-89 Z and normals are accessed only after the reconstruction
    # has been completed. They do not affect fitted coefficients or outputs.
    # -----------------------------------------------------------------------
    evaluation_available = (
        "surface_normal"
        in raster.files
        and "position"
        in raster.files
    )

    raw_height_error = None
    reconstructed_height_error = None
    normal_angle_error = None

    if evaluation_available:
        true_position = np.asarray(
            raster[
                "position"
            ],
            dtype=float,
        )

        true_normal = np.asarray(
            raster[
                "surface_normal"
            ],
            dtype=float,
        )

        raw_height_error = (
            measured_z
            - true_position[
                :,
                2
            ]
        )

        reconstructed_height_error = (
            reconstructed_z
            - true_position[
                :,
                2
            ]
        )

        normal_angle_error = np.empty(
            len(
                query_xy
            )
        )

        for index in range(
            len(
                query_xy
            )
        ):
            normal_angle_error[
                index
            ] = safe_angle_deg(
                reconstructed_normal[
                    index
                ],
                true_normal[
                    index
                ],
            )

    np.savez(
        NPZ_OUTPUT,
        position=reconstructed_position,
        surface_normal=reconstructed_normal,
        rotation=rotations,
        tool_z=tool_z,
        arc_length=arc_length,
        row_id=row_id,
        phase=phase,
        coefficients=coefficients,
        fit_rms=fit_rms,
        fit_weight=fit_weight,
        measurement_quality=base_quality,
        source_measured_position=measured_position,
        source_force_vertical=measured_force,
        desired_force=np.array(
            [
                desired_force
            ]
        ),
    )

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "index",
                "row_id",
                "phase",
                "x",
                "y",
                "z_reconstructed",
                "nx",
                "ny",
                "nz",
                "fit_rms_m",
                "fit_weight",
                "measurement_quality",
            ]
        )

        for index in range(
            len(
                query_xy
            )
        ):
            writer.writerow(
                [
                    index,
                    int(
                        row_id[
                            index
                        ]
                    ),
                    int(
                        phase[
                            index
                        ]
                    ),
                    float(
                        query_xy[
                            index,
                            0
                        ]
                    ),
                    float(
                        query_xy[
                            index,
                            1
                        ]
                    ),
                    float(
                        reconstructed_z[
                            index
                        ]
                    ),
                    float(
                        reconstructed_normal[
                            index,
                            0
                        ]
                    ),
                    float(
                        reconstructed_normal[
                            index,
                            1
                        ]
                    ),
                    float(
                        reconstructed_normal[
                            index,
                            2
                        ]
                    ),
                    float(
                        fit_rms[
                            index
                        ]
                    ),
                    float(
                        fit_weight[
                            index
                        ]
                    ),
                    float(
                        base_quality[
                            index
                        ]
                    ),
                ]
            )

    print(
        "\n========== 97B Result 97B结果 =========="
    )

    print(
        f"Reconstructed points "
        f"重建点数: "
        f"{len(reconstructed_position)}"
    )

    print(
        f"Reconstructed path length "
        f"重建路径长度: "
        f"{arc_length[-1] * 1000.0:.1f} mm"
    )

    print(
        f"Fit RMS mean/max "
        f"局部拟合RMS平均/最大: "
        f"{np.mean(fit_rms) * 1000.0:.3f} / "
        f"{np.max(fit_rms) * 1000.0:.3f} mm"
    )

    print(
        f"Measurement quality mean/min "
        f"测量质量权重平均/最小: "
        f"{np.mean(base_quality):.3f} / "
        f"{np.min(base_quality):.3f}"
    )

    print(
        f"Tool +Z vs -normal max "
        f"工具Z轴与负法向最大误差: "
        f"{np.max(tool_alignment_error):.6e} deg"
    )

    print(
        f"Rotation orthonormal max error "
        f"旋转正交最大误差: "
        f"{np.max(orthonormal_error):.3e}"
    )

    print(
        f"Rotation det max error "
        f"旋转行列式最大误差: "
        f"{np.max(determinant_error):.3e}"
    )

    if evaluation_available:
        print(
            "\nEvaluation only 仅评估，未参与重建:"
        )

        print(
            f"Raw height mean/max abs error "
            f"97A原始高度平均/最大绝对误差: "
            f"{np.mean(np.abs(raw_height_error)) * 1000.0:.3f} / "
            f"{np.max(np.abs(raw_height_error)) * 1000.0:.3f} mm"
        )

        print(
            f"Reconstructed height mean/RMS/max abs error "
            f"重建高度平均/RMS/最大绝对误差: "
            f"{np.mean(np.abs(reconstructed_height_error)) * 1000.0:.3f} / "
            f"{np.sqrt(np.mean(reconstructed_height_error**2)) * 1000.0:.3f} / "
            f"{np.max(np.abs(reconstructed_height_error)) * 1000.0:.3f} mm"
        )

        print(
            f"Reconstructed normal mean/max error "
            f"重建法向平均/最大误差: "
            f"{np.mean(normal_angle_error):.3f} / "
            f"{np.max(normal_angle_error):.3f} deg"
        )

    print(
        f"Saved NPZ 已保存NPZ: "
        f"{NPZ_OUTPUT.name}"
    )

    print(
        f"Saved CSV 已保存CSV: "
        f"{CSV_OUTPUT.name}"
    )

    print(
        "Result 结果: PASS if reconstruction metrics are acceptable "
        "若重建误差合理则通过"
    )


if __name__ == "__main__":
    main()
