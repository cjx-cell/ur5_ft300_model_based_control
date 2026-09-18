#!/usr/bin/env python3
"""92 - Static visual check of the UR5 inspection workcell.

No physics is advanced here.  The viewer shows the exact home configuration so
we can inspect robot placement, probe orientation, workpiece placement and the
planned scan-area corner markers without introducing controller dynamics.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "ur5_ft300_inspection_workcell.xml"


def get_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Missing model object 模型对象不存在: {name}")
    return idx


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing ur5_ft300_inspection_workcell.xml. "
            "Run 92_build_ur5_inspection_workcell.py first."
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    home_id = get_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    tip_site_id = get_id(
        model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip_site"
    )
    ft_site_id = get_id(
        model, mujoco.mjtObj.mjOBJ_SITE, "ft300_site"
    )
    workpiece_id = get_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "inspection_workpiece"
    )

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    tcp = data.site_xpos[tip_site_id].copy()
    ft = data.site_xpos[ft_site_id].copy()
    probe_length = float(np.linalg.norm(tcp - ft))

    workpiece_center = data.geom_xpos[workpiece_id].copy()
    workpiece_top = (
        workpiece_center[2]
        + model.geom_size[workpiece_id, 2]
    )

    print("========== 92 UR5 Viewer 92 UR5场景可视化 ==========")
    print("Robot 机器人: UR5 + FT300 + inspection probe")
    print(
        f"FT300-to-TCP distance FT300到TCP距离: "
        f"{probe_length * 1000:.3f} mm"
    )
    print(
        f"Initial TCP clearance 初始TCP距工件顶部: "
        f"{(tcp[2] - workpiece_top) * 1000:.1f} mm"
    )
    print("Green points 绿色点: same 200 x 120 mm scan-area corners")
    print("Viewer is static 可视化为静态，不推进动力学")
    print("Close viewer to exit 关闭窗口结束")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = np.array([0.08, -0.20, 0.83])
        viewer.cam.distance = 2.00
        viewer.cam.azimuth = 142.0
        viewer.cam.elevation = -23.0

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
