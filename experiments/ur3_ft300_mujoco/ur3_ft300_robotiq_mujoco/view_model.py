#!/usr/bin/env python3
"""Open the converted model and hold the arm near its Gazebo home pose."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ur3_ft300_robotiq_force_control.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

q_des = data.qpos[:6].copy()
kp = np.array([80.0, 80.0, 50.0, 20.0, 15.0, 8.0])
kd = np.array([12.0, 12.0, 8.0, 3.0, 2.0, 1.0])
limits = np.array([56.0, 56.0, 28.0, 12.0, 12.0, 12.0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        error = np.arctan2(
            np.sin(q_des - data.qpos[:6]),
            np.cos(q_des - data.qpos[:6]),
        )
        data.ctrl[:6] = np.clip(
            kp * error - kd * data.qvel[:6] + data.qfrc_bias[:6],
            -limits,
            limits,
        )
        data.ctrl[6] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
