import mujoco
import mujoco.viewer
import time
import numpy as np


# ============================================================
# 1. 用字符串建立一个非常简单的 MuJoCo 模型
# ============================================================

xml = """
<mujoco model="pendulum">

    <option timestep="0.001" gravity="0 0 -9.81"/>

    <worldbody>

        <!-- 固定支撑 -->
        <body name="pendulum" pos="0 0 1">

            <!--
            hinge joint：
            只有一个旋转自由度

            axis="0 1 0"
            表示绕 Y 轴旋转
            -->
            <joint
                name="joint1"
                type="hinge"
                axis="0 1 0"
            />

            <!--
            一根长度 1 m 左右的杆
            -->
            <geom
                type="capsule"
                fromto="0 0 0 0 0 -1"
                size="0.05"
                mass="1"
            />

        </body>

    </worldbody>

</mujoco>
"""


# ============================================================
# 2. 创建 Model
# ============================================================

# MjModel：
#
# 类似 Pinocchio 的 model，
#
# 描述：
# - body
# - joint
# - mass
# - inertia
# - actuator
# - sensor
# - contact
#
model = mujoco.MjModel.from_xml_string(xml)


# ============================================================
# 3. 创建 Data
# ============================================================

# MjData：
#
# 保存仿真当前状态：
#
# qpos
# qvel
# qacc
# ctrl
# contact
# ...
data = mujoco.MjData(model)


print("nq =", model.nq)
print("nv =", model.nv)

print("qpos =", data.qpos)
print("qvel =", data.qvel)
# ============================================================
# 设置初始关节角
# ============================================================

# qpos[0] 就是这个 hinge joint 的角度
#
# 设置成 45 degree
data.qpos[0] = np.deg2rad(90.0)

# 根据新的 qpos 更新所有几何/运动学量
mujoco.mj_forward(
    model,
    data
)

# ============================================================
# 4. 启动 Viewer
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        # -----------------------------------------------
        # MuJoCo 做一次物理积分
        # -----------------------------------------------
        #
        # 输入当前：
        #
        # q
        # dq
        # force / torque
        #
        # MuJoCo 内部计算动力学、重力、碰撞等，
        # 然后推进一个 timestep。
        mujoco.mj_step(
            model,
            data
        )


        # Viewer 更新显示
        viewer.sync()


        # 只是为了让实际播放速度不要太快
        time.sleep(
            model.opt.timestep
        )