import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt


# ============================================================
# 1. 加载机器人模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

model = pin.buildModelFromUrdf(urdf_path)

# 分开建立两个 data：
#
# data_controller：
# 给控制器做 inverse dynamics
#
# data_plant：
# 给“机器人”做 forward dynamics
#
# 这样逻辑上更清楚。
data_controller = model.createData()
data_plant = model.createData()


# ============================================================
# 2. 初始状态
# ============================================================

q0 = pin.neutral(model)

theta_init = np.deg2rad([
    0.0,
    -30.0,
    30.0,
    0.0,
    0.0,
    0.0
])

q = pin.integrate(
    model,
    q0,
    theta_init
)

# 初始速度为 0
v = np.zeros(model.nv)


# ============================================================
# 3. 目标状态
# ============================================================

theta_des = np.deg2rad([
    20.0,
    -60.0,
    70.0,
    -20.0,
    30.0,
    45.0
])

q_des = pin.integrate(
    model,
    q0,
    theta_des
)

# 固定姿态目标：
#
# dq_des = 0
v_des = np.zeros(model.nv)

# ddq_des = 0
a_des = np.zeros(model.nv)


# ============================================================
# 4. Computed Torque 闭环参数
# ============================================================

# 注意这里与之前 PD 的 Kp/Kd 有一个重要区别。
#
# 之前：
#
# tau_PD = Kp * e + Kd * de
#
# Kp/Kd 直接产生“力矩”。
#
# 所以不同关节惯量不同，
# Kp/Kd 也必须跟着调整。
#
#
# 现在：
#
# a_cmd = Kp * e + Kd * de
#
# Kp/Kd 产生的是“期望加速度”。
#
# 后面的 RNEA 会根据机器人惯量，
# 自动把加速度变成合适的力矩。
#
# 因此这里可以直接按照理想二阶系统选择参数。

wn = 4.0

# 阻尼比
zeta = 1.0


# 标准二阶系统：
#
# e_ddot
# +
# 2*zeta*wn*e_dot
# +
# wn^2*e
# =
# 0
#
# 因此：
#
# Kp = wn^2
# Kd = 2*zeta*wn
Kp = wn**2

Kd = 2.0 * zeta * wn


print("Computed Torque gains:")

print("Kp =", Kp)
print("Kd =", Kd)


# ============================================================
# 5. 仿真设置
# ============================================================

dt = 0.0002

T = 3.0

num_steps = int(
    T / dt
)


# ============================================================
# 6. 电机力矩限制
# ============================================================

tau_limit = model.effortLimit.copy()

print("\nTorque limits:")
print(tau_limit)


# ============================================================
# 7. 保存实验数据
# ============================================================

time_history = []

q_history = []
error_history = []

v_history = []
tau_history = []
ddq_history = []


# ============================================================
# 8. 控制循环
# ============================================================

for step in range(num_steps):

    t = step * dt


    # --------------------------------------------------------
    # A. 位置误差
    # --------------------------------------------------------

    # difference(q, q_des)
    #
    # 得到：
    #
    # 当前姿态 -> 目标姿态
    #
    # 所需要的 6 维 tangent-space 位移。
    position_error = pin.difference(
        model,
        q,
        q_des
    )


    # --------------------------------------------------------
    # B. 速度误差
    # --------------------------------------------------------

    velocity_error = (
        v_des
        -
        v
    )


    # --------------------------------------------------------
    # C. 设计 desired acceleration
    # --------------------------------------------------------

    # Computed Torque Controller
    #
    # a_cmd =
    #
    # a_des
    # +
    # Kp * position_error
    # +
    # Kd * velocity_error
    #
    #
    # 注意：
    #
    # 这里得到的不是 torque！
    #
    # 而是：
    #
    # 我们希望机器人产生的关节加速度。
    a_cmd = (
        a_des
        +
        Kp * position_error
        +
        Kd * velocity_error
    )


    # --------------------------------------------------------
    # D. Inverse Dynamics
    # --------------------------------------------------------

    # RNEA：
    #
    # 输入：
    #
    # q
    # v
    # desired acceleration
    #
    # 输出：
    #
    # 为实现这个 acceleration
    # 所需要的 torque
    #
    #
    # 数学：
    #
    # tau =
    #
    # M(q) * a_cmd
    # +
    # h(q,v)
    tau = pin.rnea(
        model,
        data_controller,
        q,
        v,
        a_cmd
    ).copy()


    # --------------------------------------------------------
    # E. 电机力矩限制
    # --------------------------------------------------------

    valid_limit = (
        np.isfinite(tau_limit)
        &
        (tau_limit > 0)
    )

    tau[valid_limit] = np.clip(
        tau[valid_limit],
        -tau_limit[valid_limit],
        tau_limit[valid_limit]
    )


    # --------------------------------------------------------
    # F. 机器人正动力学
    # --------------------------------------------------------

    # 现在把控制器计算出来的 tau
    # 真正“施加”到机器人。
    #
    # ABA：
    #
    # q + v + tau
    #
    #        ↓
    #
    #       ddq
    ddq = pin.aba(
        model,
        data_plant,
        q,
        v,
        tau
    ).copy()

    # ============================================================
# 检查 Computed Torque 是否真正实现期望加速度
# ============================================================

    if step % 1000 == 0:

        acceleration_error = (
            ddq
            -
            a_cmd
        )

        print(
            f"t = {t:.3f} s, "
            f"|ddq - a_cmd| = "
            f"{np.linalg.norm(acceleration_error):.6e}"
        )


    # --------------------------------------------------------
    # G. 数值安全检查
    # --------------------------------------------------------

    if not np.all(np.isfinite(ddq)):

        print(
            "\nERROR: NaN/Inf detected"
        )

        print("time =", t)
        print("tau =", tau)
        print("ddq =", ddq)

        break


    # --------------------------------------------------------
    # H. 数值积分
    # --------------------------------------------------------

    # 更新 velocity
    v = (
        v
        +
        ddq * dt
    )


    # 更新 configuration
    q = pin.integrate(
        model,
        q,
        v * dt
    )


    # --------------------------------------------------------
    # I. 数据记录
    # --------------------------------------------------------

    theta_current = pin.difference(
        model,
        q0,
        q
    )

    time_history.append(t)

    q_history.append(
        theta_current.copy()
    )

    error_history.append(
        position_error.copy()
    )

    v_history.append(
        v.copy()
    )

    tau_history.append(
        tau.copy()
    )

    ddq_history.append(
        ddq.copy()
    )


# ============================================================
# 9. 转 NumPy
# ============================================================

time_history = np.array(
    time_history
)

q_history = np.array(
    q_history
)

error_history = np.array(
    error_history
)

v_history = np.array(
    v_history
)

tau_history = np.array(
    tau_history
)

ddq_history = np.array(
    ddq_history
)


# ============================================================
# 10. 最终结果
# ============================================================

final_theta = pin.difference(
    model,
    q0,
    q
)

final_error = pin.difference(
    model,
    q,
    q_des
)


print("\n========== Final Result ==========")

print("\nDesired [deg]:")
print(
    np.rad2deg(theta_des)
)

print("\nFinal [deg]:")
print(
    np.rad2deg(final_theta)
)

print("\nFinal error [deg]:")
print(
    np.rad2deg(final_error)
)

print(
    "\nMax final error [deg]:",
    np.max(
        np.abs(
            np.rad2deg(final_error)
        )
    )
)


# ============================================================
# 11. Joint position plot
# ============================================================

plt.figure()

for i in range(model.nv):

    plt.plot(
        time_history,
        np.rad2deg(
            q_history[:, i]
        ),
        label=f"joint {i + 1}"
    )

    plt.axhline(
        np.rad2deg(
            theta_des[i]
        ),
        linestyle="--"
    )

plt.xlabel("Time [s]")
plt.ylabel("Joint angle [deg]")

plt.title(
    "Computed Torque Control"
)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 12. Position error
# ============================================================

plt.figure()

for i in range(model.nv):

    plt.plot(
        time_history,
        np.rad2deg(
            error_history[:, i]
        ),
        label=f"joint {i + 1}"
    )

plt.xlabel("Time [s]")
plt.ylabel("Position error [deg]")

plt.title(
    "Computed Torque - Position Error"
)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()