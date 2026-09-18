import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt


# ============================================================
# 1. 加载机器人模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

model = pin.buildModelFromUrdf(urdf_path)

# 合法 neutral configuration
q0 = pin.neutral(model)


# ============================================================
# 2. 初始姿态
# ============================================================

theta_init = np.deg2rad([
    0.0,
    -30.0,
    30.0,
    0.0,
    0.0,
    0.0
])

q_init = pin.integrate(
    model,
    q0,
    theta_init
)


# ============================================================
# 3. 目标姿态
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

v_des = np.zeros(model.nv)

a_des = np.zeros(model.nv)


# ============================================================
# 4. 仿真参数
# ============================================================

dt = 0.0002       # 0.2 ms

T = 3.0

num_steps = int(T / dt)


# ============================================================
# 5. 电机力矩限制
# ============================================================

tau_limit = model.effortLimit.copy()

valid_limit = (
    np.isfinite(tau_limit)
    &
    (tau_limit > 0)
)


# ============================================================
# 6. 为普通 PD / PD+Gravity 整定增益
# ============================================================

# 我们依然使用初始姿态下质量矩阵对角线，
# 粗略作为各个关节的等效惯量。
data_gain = model.createData()

M_upper = pin.crba(
    model,
    data_gain,
    q_init
).copy()

M0 = (
    np.triu(M_upper)
    +
    np.triu(M_upper, 1).T
)

I_eff = np.diag(M0)


# 希望各关节大约具有相同自然频率
wn_joint = np.array([
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0
])

zeta_joint = 1.0


# ------------------------------------------------------------
# 普通关节 PD：
#
# tau = Kp*e + Kd*edot
#
# 因为它直接输出力矩，
# 增益需要考虑各关节惯量。
# ------------------------------------------------------------

Kp_joint = (
    I_eff
    *
    wn_joint**2
)

Kd_joint = (
    2.0
    *
    zeta_joint
    *
    I_eff
    *
    wn_joint
)


# ============================================================
# 7. Computed Torque 增益
# ============================================================

# Computed Torque 中：
#
# Kp / Kd 作用在期望加速度层，
#
# a_cmd = Kp*e + Kd*edot
#
# 所以不用再乘等效惯量。

wn_ct = 4.0

zeta_ct = 1.0

Kp_ct = wn_ct**2

Kd_ct = (
    2.0
    *
    zeta_ct
    *
    wn_ct
)


print("===================================")
print("PD gains")
print("===================================")

print("Kp:")
print(Kp_joint)

print("\nKd:")
print(Kd_joint)


print("\n===================================")
print("Computed Torque gains")
print("===================================")

print("Kp =", Kp_ct)
print("Kd =", Kd_ct)


# ============================================================
# 8. 单个控制器的仿真函数
# ============================================================

def simulate(controller_name):

    """
    controller_name 可以是：

        "PD"
        "PD_GRAVITY"
        "COMPUTED_TORQUE"

    返回：

        time
        joint position
        position error
        velocity
        torque
    """

    # --------------------------------------------------------
    # 每个控制器都重新从完全相同状态开始
    # --------------------------------------------------------

    q = q_init.copy()

    v = np.zeros(model.nv)


    # --------------------------------------------------------
    # 每次仿真单独创建 data
    #
    # 避免不同控制器之间共享缓存，
    # 让实验逻辑更干净。
    # --------------------------------------------------------

    data_controller = model.createData()

    data_plant = model.createData()


    # --------------------------------------------------------
    # 数据记录
    # --------------------------------------------------------

    time_history = []

    q_history = []

    error_history = []

    velocity_history = []

    torque_history = []

    acceleration_history = []


    # ========================================================
    # 控制循环
    # ========================================================

    for step in range(num_steps):

        t = step * dt


        # ----------------------------------------------------
        # A. 当前位置误差
        # ----------------------------------------------------

        # 从当前 q 到目标 q_des 的
        # tangent-space configuration error。
        position_error = pin.difference(
            model,
            q,
            q_des
        )


        # ----------------------------------------------------
        # B. 当前速度误差
        # ----------------------------------------------------

        velocity_error = (
            v_des
            -
            v
        )


        # ====================================================
        # C. 根据不同控制器计算 tau
        # ====================================================


        # ----------------------------------------------------
        # Controller 1：纯 PD
        # ----------------------------------------------------

        if controller_name == "PD":

            # 没有任何动力学补偿。
            #
            # tau =
            #
            # Kp e
            # +
            # Kd edot
            tau = (
                Kp_joint * position_error
                +
                Kd_joint * velocity_error
            )


        # ----------------------------------------------------
        # Controller 2：PD + Gravity
        # ----------------------------------------------------

        elif controller_name == "PD_GRAVITY":

            # 先算普通 PD
            tau_pd = (
                Kp_joint * position_error
                +
                Kd_joint * velocity_error
            )


            # 计算当前姿态的重力补偿
            gravity = pin.computeGeneralizedGravity(
                model,
                data_controller,
                q
            ).copy()


            # feedback + feedforward
            tau = (
                tau_pd
                +
                gravity
            )


        # ----------------------------------------------------
        # Controller 3：Computed Torque
        # ----------------------------------------------------

        elif controller_name == "COMPUTED_TORQUE":

            # 首先设计希望产生的 acceleration
            #
            # a_cmd =
            #
            # ddq_des
            # +
            # Kp e
            # +
            # Kd edot
            a_cmd = (
                a_des
                +
                Kp_ct * position_error
                +
                Kd_ct * velocity_error
            )


            # RNEA 做 inverse dynamics
            #
            # tau =
            #
            # M(q) a_cmd
            # +
            # h(q,v)
            tau = pin.rnea(
                model,
                data_controller,
                q,
                v,
                a_cmd
            ).copy()


        else:

            raise ValueError(
                f"Unknown controller: {controller_name}"
            )


        # ====================================================
        # D. 电机力矩饱和
        # ====================================================

        tau[valid_limit] = np.clip(
            tau[valid_limit],
            -tau_limit[valid_limit],
            tau_limit[valid_limit]
        )


        # ====================================================
        # E. Plant：ABA 正动力学
        # ====================================================

        # 注意：
        #
        # 不论使用哪种控制器，
        # 被控机器人都完全相同。
        #
        # ABA：
        #
        # q, v, tau
        #
        #    ↓
        #
        # ddq
        ddq = pin.aba(
            model,
            data_plant,
            q,
            v,
            tau
        ).copy()


        # ====================================================
        # F. 数值安全检查
        # ====================================================

        if not np.all(
            np.isfinite(ddq)
        ):

            print(
                controller_name,
                "failed at",
                t,
                "seconds"
            )

            break


        # ====================================================
        # G. 积分速度
        # ====================================================

        v = (
            v
            +
            ddq * dt
        )


        # ====================================================
        # H. 积分 position/configuration
        # ====================================================

        q = pin.integrate(
            model,
            q,
            v * dt
        )


        # ====================================================
        # I. 保存数据
        # ====================================================

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

        velocity_history.append(
            v.copy()
        )

        torque_history.append(
            tau.copy()
        )

        acceleration_history.append(
            ddq.copy()
        )


    # ========================================================
    # 转 NumPy array
    # ========================================================

    return {

        "time":
            np.array(time_history),

        "q":
            np.array(q_history),

        "error":
            np.array(error_history),

        "velocity":
            np.array(velocity_history),

        "torque":
            np.array(torque_history),

        "acceleration":
            np.array(acceleration_history),

        "final_q":
            q.copy()
    }


# ============================================================
# 9. 运行三个控制器
# ============================================================

print("\nRunning PD...")

result_pd = simulate(
    "PD"
)


print("Running PD + Gravity...")

result_pdg = simulate(
    "PD_GRAVITY"
)


print("Running Computed Torque...")

result_ct = simulate(
    "COMPUTED_TORQUE"
)


# ============================================================
# 10. 性能指标函数
# ============================================================

def calculate_metrics(result):

    error = result["error"]

    time = result["time"]

    torque = result["torque"]


    # --------------------------------------------------------
    # 最终误差
    # --------------------------------------------------------

    final_q = result["final_q"]

    final_error = pin.difference(
        model,
        final_q,
        q_des
    )

    final_error_deg = np.rad2deg(
        final_error
    )


    # --------------------------------------------------------
    # 最大最终关节误差
    # --------------------------------------------------------

    max_final_error = np.max(
        np.abs(
            final_error_deg
        )
    )


    # --------------------------------------------------------
    # 整个运动过程的 RMS error
    # --------------------------------------------------------

    # 先把所有关节所有时刻误差平方，
    # 然后平均，再开根号。
    rms_error = np.sqrt(
        np.mean(
            np.rad2deg(error)**2
        )
    )


    # --------------------------------------------------------
    # 最大控制力矩
    # --------------------------------------------------------

    max_torque = np.max(
        np.abs(torque)
    )


    # --------------------------------------------------------
    # Settling time
    # --------------------------------------------------------

    # 定义：
    #
    # 所有关节误差进入 ±1 degree，
    # 并且此后一直保持在 ±1 degree 内。
    threshold = 1.0

    error_deg = np.abs(
        np.rad2deg(error)
    )

    settling_time = np.nan


    # 从前往后检查每一个时刻。
    for i in range(len(time)):

        # 从 i 时刻一直到仿真结束，
        # 所有关节是否始终 < 1 deg
        if np.all(
            error_deg[i:] < threshold
        ):

            settling_time = time[i]

            break


    return {

        "final_error":
            final_error_deg,

        "max_final_error":
            max_final_error,

        "rms_error":
            rms_error,

        "max_torque":
            max_torque,

        "settling_time":
            settling_time
    }


# ============================================================
# 11. 计算指标
# ============================================================

metrics_pd = calculate_metrics(
    result_pd
)

metrics_pdg = calculate_metrics(
    result_pdg
)

metrics_ct = calculate_metrics(
    result_ct
)


# ============================================================
# 12. 打印结果
# ============================================================

def print_metrics(name, metrics):

    print(
        "\n==================================="
    )

    print(name)

    print(
        "==================================="
    )

    print(
        "Final error [deg]:"
    )

    print(
        metrics["final_error"]
    )

    print(
        "Max final error [deg]:",
        metrics["max_final_error"]
    )

    print(
        "RMS error [deg]:",
        metrics["rms_error"]
    )

    print(
        "Max torque [N*m]:",
        metrics["max_torque"]
    )

    print(
        "Settling time [s]:",
        metrics["settling_time"]
    )


print_metrics(
    "PD",
    metrics_pd
)

print_metrics(
    "PD + Gravity",
    metrics_pdg
)

print_metrics(
    "Computed Torque",
    metrics_ct
)


# ============================================================
# 13. 图 1：Joint 2 position comparison
# ============================================================

# 先只选一个代表性关节，
# 不然 18 条曲线挤在一起很难看。
joint_index = 1       # joint 2


plt.figure()

plt.plot(
    result_pd["time"],
    np.rad2deg(
        result_pd["q"][:, joint_index]
    ),
    label="PD"
)

plt.plot(
    result_pdg["time"],
    np.rad2deg(
        result_pdg["q"][:, joint_index]
    ),
    label="PD + Gravity"
)

plt.plot(
    result_ct["time"],
    np.rad2deg(
        result_ct["q"][:, joint_index]
    ),
    label="Computed Torque"
)

plt.axhline(
    np.rad2deg(
        theta_des[joint_index]
    ),
    linestyle="--",
    label="Desired"
)

plt.xlabel("Time [s]")

plt.ylabel(
    "Joint 2 angle [deg]"
)

plt.title(
    "Joint 2 Position Tracking"
)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 14. 图 2：Joint 2 error comparison
# ============================================================

plt.figure()

plt.plot(
    result_pd["time"],
    np.rad2deg(
        result_pd["error"][:, joint_index]
    ),
    label="PD"
)

plt.plot(
    result_pdg["time"],
    np.rad2deg(
        result_pdg["error"][:, joint_index]
    ),
    label="PD + Gravity"
)

plt.plot(
    result_ct["time"],
    np.rad2deg(
        result_ct["error"][:, joint_index]
    ),
    label="Computed Torque"
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("Time [s]")

plt.ylabel(
    "Joint 2 error [deg]"
)

plt.title(
    "Joint 2 Position Error"
)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 15. 图 3：Joint 2 torque comparison
# ============================================================

plt.figure()

plt.plot(
    result_pd["time"],
    result_pd["torque"][:, joint_index],
    label="PD"
)

plt.plot(
    result_pdg["time"],
    result_pdg["torque"][:, joint_index],
    label="PD + Gravity"
)

plt.plot(
    result_ct["time"],
    result_ct["torque"][:, joint_index],
    label="Computed Torque"
)

plt.xlabel("Time [s]")

plt.ylabel(
    "Joint 2 torque [N*m]"
)

plt.title(
    "Joint 2 Control Torque"
)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()