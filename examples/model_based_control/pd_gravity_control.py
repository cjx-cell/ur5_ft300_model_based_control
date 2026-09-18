import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt


# ============================================================
# 1. 加载机器人动力学模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

# model：
# 机器人固定参数
#
# 包括：
# - joint
# - link
# - mass
# - inertia
# - COM
# - kinematic tree
model = pin.buildModelFromUrdf(urdf_path)

# data：
# Pinocchio 的计算缓存
data = model.createData()


# ============================================================
# 2. 初始姿态
# ============================================================

# 因为你的机器人存在 continuous joint：
#
# nq = 7
# nv = 6
#
# 因此不能简单：
#
# q = np.zeros(7)
#
# 使用合法的 neutral configuration。
q0 = pin.neutral(model)


# 初始六个实际关节角
theta_init = np.deg2rad([
    0.0,
    -30.0,
    30.0,
    0.0,
    0.0,
    0.0
])

# 从 neutral configuration 出发，
# 通过 6 维 tangent vector 得到合法 q。
q = pin.integrate(
    model,
    q0,
    theta_init
)


# ============================================================
# 3. 初始关节速度
# ============================================================

# 机器人一开始静止。
#
# v = dq
#
# nv = 6
v = np.zeros(model.nv)


# ============================================================
# 4. 设置目标关节姿态
# ============================================================

theta_des = np.deg2rad([
    20.0,
    -60.0,
    70.0,
    -20.0,
    30.0,
    45.0
])

# 同样转换为 Pinocchio 合法 configuration。
q_des = pin.integrate(
    model,
    q0,
    theta_des
)


# 目标速度为 0。
#
# 即我们希望：
#
# 机器人移动到 q_des 后停下来。
v_des = np.zeros(model.nv)


# ============================================================
# 5. 根据机器人惯量选择 PD 增益
# ============================================================

# 先计算初始姿态下的质量矩阵 M(q0)
M0_upper = pin.crba(
    model,
    data,
    q
).copy()

# 补成完整对称矩阵
M0 = (
    np.triu(M0_upper)
    +
    np.triu(M0_upper, 1).T
)

# 取对角线：
#
# 可以粗略理解为每个关节当前的“等效惯量”
I_eff = np.diag(M0)

print("Initial mass matrix diagonal:")
print(I_eff)


# ============================================================
# 希望闭环响应速度
# ============================================================

# natural frequency
#
# 数值越大：
# 控制器越快、越硬
#
# 我们第一次先保守一点
wn = np.array([
    4.0,
    4.0,
    4.0,
    5.0,
    5.0,
    5.0
])


# damping ratio
#
# zeta = 1：
# 临界阻尼附近
#
# 不追求振荡，第一次实验最合适
zeta = 1.0


# ============================================================
# 根据二阶系统公式计算增益
# ============================================================

# Kp = I * wn^2
Kp = I_eff * wn**2

# Kd = 2*zeta*I*wn
Kd = 2.0 * zeta * I_eff * wn


print("\nKp:")
print(Kp)

print("\nKd:")
print(Kd)


# ============================================================
# 6. 仿真参数
# ============================================================

# 时间步长：
#
# 0.2 ms
dt = 0.0002


# 总仿真时间：
#
# 4 seconds
T = 3.0


# 一共要计算多少次控制循环
num_steps = int(T / dt)


# ============================================================
# 7. 用来保存数据
# ============================================================

time_history = []

q_history = []
error_history = []
tau_history = []

velocity_history = []
ddq_history = []
gravity_history = []

# ============================================================
# 电机力矩限制
# ============================================================

tau_limit = model.effortLimit.copy()

print("\nTorque limits from URDF:")
print(tau_limit)
# ============================================================
# 8. 正式进入控制循环
# ============================================================

for step in range(num_steps):

    # 当前时间
    t = step * dt


    # --------------------------------------------------------
    # A. 计算 configuration error
    # --------------------------------------------------------

    # 这里非常重要。
    #
    # 对普通六轴机器人，我们很容易想到：
    #
    # error = q_des - q
    #
    # 但是你的：
    #
    # q_des.shape = (7,)
    # q.shape     = (7,)
    #
    # 更重要的是 continuous joint 使用：
    #
    # [cos(theta), sin(theta)]
    #
    # 所以不能简单用 configuration 做普通减法。
    #
    # Pinocchio 提供：
    #
    # difference(model, q, q_des)
    #
    # 它计算：
    #
    # 从当前 q 到目标 q_des
    # 需要沿 tangent space 走多少。
    #
    # 输出维度：
    #
    # nv = 6
    #
    # 正好得到六个自由度的误差。
    position_error = pin.difference(
        model,
        q,
        q_des
    )


    # --------------------------------------------------------
    # B. 速度误差
    # --------------------------------------------------------

    # 目标速度：
    #
    # v_des = 0
    #
    # 当前速度：
    #
    # v
    #
    # 因此：
    #
    # velocity_error = v_des - v
    velocity_error = (
        v_des
        -
        v
    )


    # --------------------------------------------------------
    # C. PD Controller
    # --------------------------------------------------------

    # Element-wise multiplication：
    #
    # Kp * position_error
    #
    # 意味着六个关节分别使用自己的增益。
    #
    # tau_PD =
    #
    # Kp * e
    # +
    # Kd * de
    tau_pd = (
        Kp * position_error
        +
        Kd * velocity_error
    )


    # --------------------------------------------------------
    # D. Gravity Compensation
    # --------------------------------------------------------

    # 根据当前姿态 q
    #
    # 实时计算：
    #
    # g(q)
    gravity = pin.computeGeneralizedGravity(
        model,
        data,
        q
    ).copy()


        # --------------------------------------------------------
    # E. 最终控制力矩
    # --------------------------------------------------------

    # PD feedback + gravity compensation
    tau = (
        tau_pd
        +
        gravity
    )


    # --------------------------------------------------------
    # F. 力矩饱和
    # --------------------------------------------------------

    # URDF 中给出了每个关节的 effort limit。
    #
    # 真实电机不可能输出无限大的力矩，
    # 所以把控制器输出限制到：
    #
    # -tau_limit <= tau <= tau_limit
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
    # G. 把 tau 送入机器人动力学
    # --------------------------------------------------------

    # ABA 求正动力学：
    #
    # M(q) ddq + h(q,v) = tau
    #
    # 输入：
    # q   当前姿态
    # v   当前关节速度
    # tau 当前控制力矩
    #
    # 输出：
    # ddq 当前关节加速度
    ddq = pin.aba(
        model,
        data,
        q,
        v,
        tau
    ).copy()


    # --------------------------------------------------------
    # H. 数值安全检查
    # --------------------------------------------------------

    # 如果出现 NaN 或 Inf，
    # 说明数值仿真已经彻底失效。
    if not np.all(np.isfinite(ddq)):

        print("\nERROR: ddq contains NaN or Inf")
        print("time =", t)
        print("q =", q)
        print("v =", v)
        print("tau =", tau)
        print("ddq =", ddq)

        break


    # 如果还没有 NaN，
    # 但加速度已经夸张到 1e5 rad/s^2，
    # 也提前结束，方便定位问题。
    if np.max(np.abs(ddq)) > 1e5:

        print("\nWARNING: dynamics is diverging")
        print("time =", t)

        print(
            "max |error| =",
            np.max(np.abs(position_error))
        )

        print(
            "max |velocity| =",
            np.max(np.abs(v))
        )

        print(
            "max |tau| =",
            np.max(np.abs(tau))
        )

        print(
            "max |ddq| =",
            np.max(np.abs(ddq))
        )

        break


    # --------------------------------------------------------
    # I. 数值积分：更新速度
    # --------------------------------------------------------

    # Euler integration：
    #
    # v(k+1) =
    # v(k) + ddq(k) * dt
    v = (
        v
        +
        ddq * dt
    )


    # --------------------------------------------------------
    # J. 数值积分：更新 configuration
    # --------------------------------------------------------

    # 因为你的：
    #
    # nq = 7
    # nv = 6
    #
    # 不能直接写：
    #
    # q = q + v * dt
    #
    # 使用 Pinocchio integrate
    # 在 configuration manifold 上更新 q。
    q = pin.integrate(
        model,
        q,
        v * dt
    )


    # --------------------------------------------------------
    # K. 保存数据
    # --------------------------------------------------------

    time_history.append(t)

    # 把当前 configuration 转回
    # 6 维“关节角变化”形式，
    # 方便画图。
    theta_current = pin.difference(
        model,
        q0,
        q
    )

    q_history.append(
        theta_current.copy()
    )

    error_history.append(
        position_error.copy()
    )

    tau_history.append(
        tau.copy()
    )

    velocity_history.append(
        v.copy()
    )

    ddq_history.append(
        ddq.copy()
    )

    gravity_history.append(
        gravity.copy()
    )


# ============================================================
# 9. 转成 NumPy 数组
# ============================================================

time_history = np.array(time_history)

q_history = np.array(q_history)

error_history = np.array(error_history)

tau_history = np.array(tau_history)


# ============================================================
# 10. 打印最终结果
# ============================================================

print("========== Final Result ==========")

print("\nDesired joint angles [deg]:")
print(
    np.rad2deg(theta_des)
)

print("\nFinal joint angles [deg]:")

final_theta = pin.difference(
    model,
    q0,
    q
)

print(
    np.rad2deg(final_theta)
)


print("\nFinal error [deg]:")

final_error = pin.difference(
    model,
    q,
    q_des
)

print(
    np.rad2deg(final_error)
)


# ============================================================
# 11. 绘制关节轨迹
# ============================================================

plt.figure()

for i in range(model.nv):

    plt.plot(
        time_history,
        np.rad2deg(q_history[:, i]),
        label=f"joint {i + 1}"
    )

    # 画目标值：
    #
    # 每个 joint 一条水平虚线
    plt.axhline(
        np.rad2deg(theta_des[i]),
        linestyle="--"
    )


plt.xlabel("Time [s]")
plt.ylabel("Joint angle [deg]")

plt.title(
    "PD + Gravity Compensation"
)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 12. 绘制位置误差
# ============================================================

plt.figure()

for i in range(model.nv):

    plt.plot(
        time_history,
        np.rad2deg(error_history[:, i]),
        label=f"joint {i + 1}"
    )

plt.xlabel("Time [s]")
plt.ylabel("Position error [deg]")

plt.title(
    "Joint Position Error"
)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()