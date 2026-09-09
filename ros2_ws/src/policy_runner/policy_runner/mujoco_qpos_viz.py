"""Push an ACT action (and optionally the real robot's observed qpos, as a shadow
overlay) into a MuJoCo model and render it.

Joint-space checkpoints (set_qpos): straight into qpos, FK only -- no actuators, no
`mj_step`, no physics.

task_space checkpoints (set_task_space_target): the main robot has no arm actuators of
its own in this MJCF, just a `mocap` body weld-constrained to RHand_PALM_LINK, so there's
no way to show the policy's predicted pose without actually stepping physics and letting
the weld's constraint solve pull the arm to it (no analytic IK). Wrist (WRZ/WRY) has no
actuator either -- poked into qpos every substep, since the weld alone leaves it
underdetermined. The 16 finger joints and the shadow robot's 24 joints DO have their own
position actuators, fed via ctrl.
"""

import mujoco
import mujoco.viewer
import numpy as np

# Joint order the training qpos/action vectors are actually in: trajectories/combined/
# joint_names_json in the raw teleop trajectory.h5 (data_0901/data_0902, checked directly
# -- act/forward_kinematics.py JOINT_NAMES is the same list). CORRECTED 2026-09-08: the
# previous version of this list had the finger groups as I,M,R,T; range-checking real qpos
# columns against hmf_hand_proto5_release_right_ur7e_hand_actuator.xml's ctrlrange (thumb's
# T1Z/T1Y ranges are distinctively asymmetric vs the other three fingers' identical ranges)
# shows the actual order is T,I,M,R. The 2026-09-03 "verified via dataset_stats.pkl
# finger-joint correlations" comment this replaces was wrong -- I/M/R share identical
# ctrlranges, so that check couldn't have distinguished them from a swapped order; only the
# thumb's distinct range exposes it. This silently mislabeled hand joints on every real-robot
# UDP send and every live /joint_states reorder before this fix.
# Real robot's live /joint_states does NOT publish in this order either (WRZ/WRY come last
# there, not right after the arm) -- callers must reorder a live JointState by `name` into
# this order before using it here; act_infer_mujoco.py's InferInputNode does this; don't feed
# raw positional /joint_states arrays into set_qpos.
JOINT_ORDER = [
    "RArm_shoulder_pan_joint", "RArm_shoulder_lift_joint", "RArm_elbow_joint",
    "RArm_wrist_1_joint", "RArm_wrist_2_joint", "RArm_wrist_3_joint",
    "RHand_WRZ_joint", "RHand_WRY_joint",
    "RHand_T1Z_joint", "RHand_T1Y_joint", "RHand_T2Y_joint", "RHand_T3Y_joint",
    "RHand_I1Z_joint", "RHand_I1Y_joint", "RHand_I2Y_joint", "RHand_I3Y_joint",
    "RHand_M1Z_joint", "RHand_M1Y_joint", "RHand_M2Y_joint", "RHand_M3Y_joint",
    "RHand_R1Z_joint", "RHand_R1Y_joint", "RHand_R2Y_joint", "RHand_R3Y_joint",
]

# The scene template also carries a translucent, collision-free "Shadow_" duplicate
# of the same 24 joints (see the MJCF's shadow-robot block) used to overlay the real
# robot's observed qpos next to the policy's predicted action. Same order/caveat.
SHADOW_JOINT_ORDER = [f"Shadow_{name}" for name in JOINT_ORDER]


class MujocoQposViz:
    def __init__(self, mjcf_path, launch_viewer=True, task_space=False, control_hz=30.0, hand_joint_names=None):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        expected_nq = len(JOINT_ORDER) + len(SHADOW_JOINT_ORDER)
        if expected_nq != self.model.nq:
            raise ValueError(f"expected nq={expected_nq} (main + shadow joints) but model nq={self.model.nq}")
        self.qpos_addr = np.array([self.model.joint(name).qposadr[0] for name in JOINT_ORDER])
        self.shadow_qpos_addr = np.array([self.model.joint(name).qposadr[0] for name in SHADOW_JOINT_ORDER])
        # Per-joint (low, high) limits from the MJCF, in JOINT_ORDER -- unlimited joints get +-inf.
        self.joint_range = np.array(
            [
                self.model.joint(name).range if self.model.joint(name).limited[0] else (-np.inf, np.inf)
                for name in JOINT_ORDER
            ]
        )
        self.task_space = task_space
        if task_space:
            # hand_joint_names is policy.hand_joint_names: [WRZ, WRY, <16 finger joints>],
            # same order as JOINT_ORDER[6:] (see forward_kinematics.JOINT_NAMES).
            hand_joint_names = list(hand_joint_names)
            wrist_names, finger_names = hand_joint_names[:2], hand_joint_names[2:]
            self.wrist_qpos_addr = np.array([self.model.joint(n).qposadr[0] for n in wrist_names])
            self.wrist_dof_addr = np.array([self.model.joint(n).dofadr[0] for n in wrist_names])
            self.finger_actuator_ids = np.array([self.model.actuator(f"{n}_ctrl").id for n in finger_names])
            self.shadow_actuator_ids = np.array([self.model.actuator(f"Shadow_{n}_ctrl").id for n in JOINT_ORDER])
            self.mocap_id = self.model.body("mocap").mocapid[0]
            # Sub-step physics to fill exactly one control period, so the weld's constraint
            # solve gets the same amount of settling time regardless of control_hz.
            self.n_substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if launch_viewer else None

    def set_qpos(self, action, shadow_qpos=None):
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] != len(JOINT_ORDER):
            raise ValueError(f"expected {len(JOINT_ORDER)}-dim action, got {action.shape}")
        self.data.qpos[self.qpos_addr] = action
        if shadow_qpos is not None:
            shadow_qpos = np.asarray(shadow_qpos, dtype=np.float64)
            if shadow_qpos.shape[0] != len(SHADOW_JOINT_ORDER):
                raise ValueError(f"expected {len(SHADOW_JOINT_ORDER)}-dim shadow_qpos, got {shadow_qpos.shape}")
            self.data.qpos[self.shadow_qpos_addr] = shadow_qpos
        mujoco.mj_forward(self.model, self.data)  # kinematics only -- no mj_step, no ctrl

    def set_task_space_target(self, pos, quat_wxyz, hand, shadow_qpos):
        """task_space mode only. pos(3)/quat_wxyz(4): mocap target for the weld to
        RHand_PALM_LINK. hand(18): [WRZ, WRY, <16 finger joints>] -- WRZ/WRY have no
        actuator so they're poked into qpos every substep; the rest go through their
        position actuators. shadow_qpos(24): real observed qpos, JOINT_ORDER order --
        fed as the shadow's own ctrl (not a direct qpos poke) so it holds its pose
        correctly now that physics is being stepped."""
        hand = np.asarray(hand, dtype=np.float64)
        self.data.mocap_pos[self.mocap_id] = pos
        self.data.mocap_quat[self.mocap_id] = quat_wxyz
        self.data.ctrl[self.finger_actuator_ids] = hand[2:]
        self.data.ctrl[self.shadow_actuator_ids] = shadow_qpos
        for _ in range(self.n_substeps):
            # Re-poked every substep (not just once before the loop): with no actuator
            # of its own, dynamics would otherwise carry the wrist away from the
            # policy's commanded orientation over the course of the settle.
            self.data.qpos[self.wrist_qpos_addr] = hand[:2]
            self.data.qvel[self.wrist_dof_addr] = 0.0
            mujoco.mj_step(self.model, self.data)

    def sync(self):
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
