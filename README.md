# UR3 + FT300 Model-Based Control & Contact Manipulation

Model-based motion and force control experiments for a **UR3 manipulator + FT300 force/torque sensor + Robotiq gripper**, built with **Pinocchio**, **MuJoCo**, and **TSID/HQP**.

This repository focuses on robot kinematics, rigid-body dynamics, computed-torque control, Cartesian impedance/admittance control, force sensing and compensation, hybrid position/force control, surface-contact manipulation, robustness evaluation, and task-space inverse dynamics.

## Project Scope

```text
SE(3) geometry
    ↓
Forward kinematics
    ↓
Jacobian / wrench mapping
    ↓
Rigid-body dynamics
    ↓
Joint-space model-based control
    ↓
Cartesian pose control
    ↓
Impedance / admittance control
    ↓
FT300 force processing
    ↓
Hybrid position-force control
    ↓
Surface scanning and force-controlled replay
    ↓
Robustness evaluation
    ↓
TSID / HQP multi-task and contact-force optimization
```

The main objective is not only to implement controllers, but also to verify model consistency, isolate simulation/modeling errors, and evaluate contact-control robustness under controlled perturbations.

## Repository Structure

```text
ur3_ft300_model_based_control/
├── README.md
├── environment.yml
├── .gitignore
├── examples/
│   ├── geometry/
│   ├── forward_kinematics/
│   ├── jacobian/
│   ├── wrench_mapping/
│   ├── dynamics/
│   └── model_based_control/
├── experiments/
│   └── ur3_ft300_mujoco/
└── docs/
```

## Core Components

### Kinematics and Dynamics

The examples cover SO(3)/SE(3) transforms, forward kinematics, frame Jacobians, wrench-to-joint-torque mapping, gravity compensation, mass-matrix computation, nonlinear effects, and RNEA/CRBA/ABA.

The rigid-body dynamics are

\[
M(q)\ddot q + h(q,\dot q) = \tau .
\]

### Model-Based Motion Control

Implemented controllers include joint PD, PD + gravity compensation, computed-torque control, 3D Cartesian position control, 6D Cartesian pose control, and 3D/6D Cartesian impedance control.

The Cartesian acceleration relationship is

\[
\ddot x = J(q)\ddot q + \dot J(q,\dot q)\dot q .
\]

### FT300 Force/Torque Processing

The force-sensing pipeline includes raw wrench reading, zeroing, payload-gravity compensation, filtering, sensor/world frame transforms, and projection onto task or surface directions.

### Admittance and Hybrid Position/Force Control

Implemented experiments include 1D/6D admittance, hybrid position-force control, surface-normal force tracking, and curved-surface TCP pose + force tracking.

A representative admittance relation is

\[
M_f\ddot d_n + D_f\dot d_n = F_d - F_n .
\]

## Surface Contact and Path Learning

The contact-manipulation workflow is:

```text
Low-force scan
    ↓
Record actual TCP trajectory
    ↓
Arc-length resampling
    ↓
Local polynomial fitting
    ↓
Estimate tangent and surface normal
    ↓
Learn nominal path
    ↓
Force-controlled replay
```

Representative path-learning results:

- Raw scan samples: 1500
- Learned path points: 301
- Learned path length: 30.66 mm
- Scan-force mean error: 0.022 N
- Path mean / max error: 0.008 / 0.023 mm
- Surface-normal mean / max error: 0.10 / 0.96 deg

Representative force-controlled replay results:

- Normal-force mean / max error: **0.027 / 0.182 N**
- TCP mean / max error: **0.108 / 0.560 mm**
- Tool-to-learned-normal mean / max error: **0.057 / 0.245 deg**

## Robustness Evaluation

| Perturbation | Main observation |
|---|---|
| Workpiece normal offset +1.5 mm | Force controller compensated the offset |
| Friction coefficient 0.05 → 0.30 | Normal force remained stable; TCP / pose accuracy degraded |
| Tool radius 5.0 → 4.5 mm | Force correction compensated the geometric change |
| 0.15 mm scan-position noise | Direct differentiation failed; local quadratic fitting recovered usable normals |

The noisy-scan experiment demonstrates that numerical differentiation can amplify measurement noise; local regression was therefore used before tangent/normal estimation.

## Dynamics Debugging: MuJoCo Passive Forces

Free-space isolation tests identified a missing MuJoCo passive generalized-force term. The corrected actuator command is

\[
\tau_{act} = \tau_{target} - q_{passive}.
\]

The debugging workflow was:

```text
Observed tracking error
    ↓
Form hypothesis
    ↓
Single-variable experiment
    ↓
Isolate free-space subsystem
    ↓
Identify missing dynamics term
    ↓
Re-close loop and verify
```

## TSID / HQP Experiments

Implemented experiments:

- `80_ur3_tsid_posture_control.py` — joint-posture task
- `81_ur3_tsid_se3_control_fixed.py` — 6D SE(3) task
- `82_ur3_tsid_multitask_bounds_fixed.py` — Cartesian task + posture regularization + joint-velocity bounds
- `83_ur3_tsid_contact_force.py` — point contact + contact-force optimization + friction constraints

For the contact experiment:

```text
nVar = 15
nEq  = 3
nIn  = 5
```

Representative result:

- Desired normal force: 5.000 N
- Optimized normal force: 4.950495 N
- Max contact-point position error: 0.000000 mm
- Max contact-dynamics residual: 1.776e-15

The verified contact dynamics are

\[
M(q)\ddot q + h(q,\dot q) = \tau + J_c^T f_c .
\]

## Recommended Experiments

```text
experiments/ur3_ft300_mujoco/

12_ur3_computed_torque_5dof.py
21_ur3_cartesian_pose_computed_torque.py
31_ur3_cartesian_impedance_6d.py
61_ur3_admittance_6d.py
72_ur3_curved_surface_tcp_passive_comp.py
74_ur3_contact_path_learning.py
74_ur3_force_controlled_replay.py
75_ur3_workpiece_offset_robustness.py
75_ur3_friction_robustness.py
75_ur3_tool_radius_robustness.py
75_ur3_scan_noise_robustness_v2.py
80_ur3_tsid_posture_control.py
81_ur3_tsid_se3_control_fixed.py
82_ur3_tsid_multitask_bounds_fixed.py
83_ur3_tsid_contact_force.py
```

Other scripts are retained to document intermediate hypotheses, failed approaches, and debugging experiments.

## Environment

Tested core environment:

- Ubuntu 22.04
- Python 3.10
- Pinocchio 4.0.0
- TSID 1.10.0
- MuJoCo 3.13.0
- NumPy

Create the environment with:

```bash
conda env create -f environment.yml
conda activate ur3-control
```

If ROS 2 environment variables are already loaded in the shell, launch standalone TSID/MuJoCo experiments with:

```bash
env -u PYTHONPATH python experiments/ur3_ft300_mujoco/83_ur3_tsid_contact_force.py
```

## Notes

- This repository is a simulation and algorithm-validation project.
- TSID contact forces are model-side optimized contact forces, not direct FT300 measurements.
- Simulation gains should not be copied directly to real hardware without safety limits, actuator constraints, frequency validation, and contact testing.

## Related Project

The separate repository **`ur3_ft300_ws`** focuses on ROS 2 / Gazebo system integration, multimodal sensing, data collection, and learned manipulation policies. This repository focuses on model-based robot dynamics, force control, MuJoCo, and TSID/HQP.
