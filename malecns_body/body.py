"""Shared definition of the humanoid body.

Everything a controller needs to know about the body lives here, so that the
training environment, the closed-loop brain demo and the hardware deployment
interface (``malecns_body.deploy``) all agree on joint order, default pose,
units and limits.
"""
from __future__ import annotations

import os

import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
HUMANOID_XML = os.path.join(ASSET_DIR, "humanoid.xml")

# Actuator order (== order of joint targets sent to the body).
LEG_JOINTS = [
    "lumbar_bend", "lumbar_flex", "lumbar_twist",
    "hip_flex_l", "hip_abd_l", "hip_rot_l", "knee_l", "ankle_flex_l", "ankle_inv_l",
    "hip_flex_r", "hip_abd_r", "hip_rot_r", "knee_r", "ankle_flex_r", "ankle_inv_r",
]
ARM_JOINTS = [
    "shoulder_flex_l", "shoulder_abd_l", "shoulder_rot_l", "elbow_l",
    "shoulder_flex_r", "shoulder_abd_r", "shoulder_rot_r", "elbow_r",
]
ALL_JOINTS = LEG_JOINTS + ARM_JOINTS
N_LEG = len(LEG_JOINTS)
N_ARM = len(ARM_JOINTS)
N_ALL = len(ALL_JOINTS)

# Default posture (rad): slight crouch with flat feet, arms relaxed.
_DEFAULT = {
    "hip_flex_l": 0.15, "knee_l": 0.30, "ankle_flex_l": 0.15,
    "hip_flex_r": 0.15, "knee_r": 0.30, "ankle_flex_r": 0.15,
    "shoulder_abd_l": 0.06, "shoulder_abd_r": 0.06,
    "elbow_l": 0.25, "elbow_r": 0.25,
}
DEFAULT_POSE = np.array([_DEFAULT.get(j, 0.0) for j in ALL_JOINTS])
DEFAULT_PELVIS_HEIGHT = 0.926  # m, pelvis height in the default posture

# Hardware-facing actuator spec: PD gains used by the joint servo, peak torque
# and maximum joint speed. These are the numbers a physical body's motor
# drivers must reproduce (see deploy.py / docs/HARDWARE.md).
ACTUATOR_SPEC = {
    #  name            kp     kd    tau_max  vel_max(rad/s)
    "lumbar":       (400.0, 30.0, 300.0, 10.0),
    "hip":          (300.0, 20.0, 260.0, 18.0),
    "hip_rot":      (150.0,  8.0, 150.0, 18.0),
    "knee":         (300.0, 15.0, 300.0, 20.0),
    "ankle":        (300.0, 10.0, 200.0, 20.0),
    "ankle_inv":    (120.0,  5.0, 100.0, 20.0),
    "shoulder":     ( 60.0,  4.0,  70.0, 15.0),
    "elbow":        ( 40.0,  3.0,  45.0, 15.0),
}


def actuator_group(joint: str) -> str:
    if joint.startswith("lumbar"):
        return "lumbar"
    if joint.startswith("hip_rot"):
        return "hip_rot"
    if joint.startswith("hip"):
        return "hip"
    if joint.startswith("knee"):
        return "knee"
    if joint.startswith("ankle_inv"):
        return "ankle_inv"
    if joint.startswith("ankle"):
        return "ankle"
    if joint.startswith("shoulder"):
        return "shoulder"
    return "elbow"


VEL_LIMIT = np.array([ACTUATOR_SPEC[actuator_group(j)][3] for j in ALL_JOINTS])
TAU_LIMIT = np.array([ACTUATOR_SPEC[actuator_group(j)][2] for j in ALL_JOINTS])

CONTROL_DT = 0.02      # s, policy rate 50 Hz
SIM_DT = 0.004         # s, physics / servo rate 250 Hz
N_SUBSTEPS = int(round(CONTROL_DT / SIM_DT))
ACTION_SCALE = 0.5     # rad per unit policy action


# ----------------------------------------------------------------------------
# Gait timing (shared by the reward in training and the clock at run time)
# ----------------------------------------------------------------------------

def gait_params(vx):
    """Stride frequency (Hz) and swing fraction for a commanded speed (m/s).

    Walking (< ~1.8 m/s): human-like cadence, 40% swing => double support.
    Running (> ~2.0 m/s): higher cadence, 62% swing => flight phases.
    """
    s = np.abs(np.asarray(vx, dtype=float))
    walk_f = 0.75 + 0.2 * s
    run_f = 1.25 + 0.08 * s
    w = np.clip((s - 1.7) / 0.3, 0.0, 1.0)          # 0 = walk, 1 = run
    freq = (1 - w) * walk_f + w * run_f
    swing = (1 - w) * 0.40 + w * 0.62
    return freq, swing, w


JUMP_DURATION = 0.9            # s, whole jump manoeuvre
JUMP_TAKEOFF = 0.30            # fraction of jump at which flight should start
JUMP_LANDING = 0.62            # fraction at which feet should be back down


# ----------------------------------------------------------------------------
# Leg trajectory generator (rhythm), modulated by the learned policy
# ----------------------------------------------------------------------------
# The policy outputs *corrections* on top of this clock-driven rhythm
# ("Policies Modulating Trajectory Generators", Iscen et al. 2018). The
# generator gives the basic stepping / jumping rhythm; the policy learns the
# balance, speed, turning and landing feedback. It is part of the deployed
# controller.

_J = {n: i for i, n in enumerate(LEG_JOINTS)}
_JUMP_KEYS = np.array([0.0, 0.15, 0.27, 0.33, 0.45, 0.58, 0.70, 0.85, 1.0])
_JUMP_HIP = np.array([0.0, 0.70, 0.75, -0.15, 0.90, 0.40, 0.60, 0.30, 0.0])
_JUMP_KNEE = np.array([0.0, 1.10, 1.20, -0.28, 1.40, 0.50, 0.90, 0.40, 0.0])
_JUMP_PUSH = np.array([0.0, 0.00, 0.00, -0.45, 0.00, 0.00, 0.00, 0.00, 0.0])
_JUMP_TRUNK = np.array([0.0, 0.35, 0.40, 0.00, 0.20, 0.10, 0.25, 0.10, 0.0])


def _leg_rhythm(p, s, H, K, Kst, P):
    """Hip, knee, ankle offsets for one leg at phase p with swing fraction s."""
    in_sw = p < s
    u_sw = np.clip(p / s, 0.0, 1.0)
    u_st = np.clip((p - s) / (1.0 - s), 0.0, 1.0)
    hip = np.where(in_sw, -H * np.cos(np.pi * u_sw), H * np.cos(np.pi * u_st))
    knee = np.where(in_sw, K * np.sin(np.pi * u_sw), Kst * np.sin(np.pi * u_st))
    push = np.where(in_sw, 0.12 * np.sin(np.pi * u_sw), -P * np.clip(np.sin(np.pi * (2 * u_st - 1.0)), 0, 1))
    # stance: keep the sole parallel to the ground; swing: partial compensation
    ankle = np.where(in_sw, 0.5 * (knee - hip), knee - hip) + push
    return hip, knee, ankle


def leg_rhythm(phase, vx, yaw, swing, run_w, stand, in_jump, jump_frac):
    """Feed-forward leg/trunk offsets (N, 15) added to DEFAULT_POSE[:15]."""
    phase = np.asarray(phase, dtype=float)
    n = phase.shape[0]
    s_abs = np.abs(vx)
    step = np.where(stand, 0.0, 1.0)
    H = np.sign(vx) * (0.05 + 0.13 * np.minimum(s_abs, 3.5)) * step
    K = (0.45 + 0.25 * run_w + 0.08 * np.minimum(s_abs, 3.5)) * step
    Kst = (0.08 + 0.25 * run_w) * step
    P = (0.15 + 0.2 * run_w) * np.clip(s_abs, 0, 1) * step
    out = np.zeros((n, N_LEG))
    for leg, off in (("l", 0.0), ("r", 0.5)):
        h, k, a = _leg_rhythm((phase + off) % 1.0, swing, H, K, Kst, P)
        out[:, _J["hip_flex_" + leg]] = h
        out[:, _J["knee_" + leg]] = k
        out[:, _J["ankle_flex_" + leg]] = a
    out[:, _J["lumbar_flex"]] = 0.06 * run_w + 0.03 * np.clip(vx, 0, 1)
    # jump manoeuvre (both legs together): crouch, push, tuck, land, absorb
    jf = np.asarray(jump_frac, dtype=float)
    jh, jk = np.interp(jf, _JUMP_KEYS, _JUMP_HIP), np.interp(jf, _JUMP_KEYS, _JUMP_KNEE)
    ja = jk - jh + np.interp(jf, _JUMP_KEYS, _JUMP_PUSH)
    jt = np.interp(jf, _JUMP_KEYS, _JUMP_TRUNK)
    ij = np.asarray(in_jump, dtype=bool)
    for leg in ("l", "r"):
        out[ij, _J["hip_flex_" + leg]] = jh[ij]
        out[ij, _J["knee_" + leg]] = jk[ij]
        out[ij, _J["ankle_flex_" + leg]] = ja[ij]
    out[ij, _J["lumbar_flex"]] = jt[ij]
    return out


# ----------------------------------------------------------------------------
# Arm motor programs (not learned: gait-coupled swing, jump swing, feeding)
# ----------------------------------------------------------------------------

def arm_swing_targets(phase, vx, run_w, swing_frac):
    """Counter-phase arm swing coupled to the leg clock. Returns (N, 8)."""
    phase = np.asarray(phase, dtype=float)
    n = phase.shape[0]
    amp = np.clip(0.15 + 0.18 * np.abs(vx), 0.0, 0.75)
    # left arm most backward at left foot-strike (phase == swing_frac)
    c = np.cos(2 * np.pi * (phase - swing_frac))
    flex_l = -amp * c
    flex_r = amp * c
    elbow = 0.25 + 1.2 * run_w
    out = np.zeros((n, N_ARM))
    out[:, 0] = flex_l
    out[:, 1] = 0.06 + 0.1 * run_w
    out[:, 3] = elbow + 0.15 * amp * (1 - c)
    out[:, 4] = flex_r
    out[:, 5] = 0.06 + 0.1 * run_w
    out[:, 7] = elbow + 0.15 * amp * (1 + c)
    return out


def arm_jump_targets(jump_frac):
    """Arm swing that accompanies a jump: back during crouch, up at take-off."""
    f = np.asarray(jump_frac, dtype=float)
    n = f.shape[0]
    flex = np.interp(f, [0.0, 0.18, 0.32, 0.55, 0.85, 1.0], [0.0, -0.7, 1.5, 1.2, 0.2, 0.0])
    elbow = np.interp(f, [0.0, 0.18, 0.32, 0.6, 1.0], [0.25, 0.3, 0.4, 0.6, 0.25])
    out = np.zeros((n, N_ARM))
    out[:, 0] = out[:, 4] = flex
    out[:, 1] = out[:, 5] = 0.15
    out[:, 3] = out[:, 7] = elbow
    return out


# Hand-to-mouth posture (both hands bring food to the mouth).
FEED_POSE = np.array([1.231, -0.262, -0.442, 2.052, 1.231, -0.262, -0.442, 2.052])  # solved: hands 8 cm in front of the mouth
DEFAULT_ARMS = DEFAULT_POSE[N_LEG:]


def arm_feed_targets(level):
    """Blend from relaxed arms to the hand-to-mouth feeding posture."""
    level = np.clip(np.asarray(level, dtype=float), 0.0, 1.0)[:, None]
    return (1 - level) * DEFAULT_ARMS[None] + level * FEED_POSE[None]
