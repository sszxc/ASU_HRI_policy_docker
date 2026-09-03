"""Push an ACT action straight into a MuJoCo model's qpos and render it -- no
actuators, no `mj_step`, no physics. Forward-kinematics-only visualization.
"""

import mujoco
import mujoco.viewer
import numpy as np

# Joint order = the arm+hand actuator include order in
# hmf_hand_proto5_release_right_ur7e_{arm,hand}_actuator.xml. Verified against the
# template scene: 24 hinge joints, qpos is a plain 0..23 layout in this order (no
# other joints in the scene). This must match the order the real robot's
# /joint_states (and thus the training qpos/action vectors) are in -- true by
# construction here (it's this robot's own joint list) but not cross-checked against
# a live /joint_states message; if the rendered pose looks wrong, check this first.
JOINT_ORDER = [
    "RArm_shoulder_pan_joint", "RArm_shoulder_lift_joint", "RArm_elbow_joint",
    "RArm_wrist_1_joint", "RArm_wrist_2_joint", "RArm_wrist_3_joint",
    "RHand_WRZ_joint", "RHand_WRY_joint",
    "RHand_I1Z_joint", "RHand_I1Y_joint", "RHand_I2Y_joint", "RHand_I3Y_joint",
    "RHand_M1Z_joint", "RHand_M1Y_joint", "RHand_M2Y_joint", "RHand_M3Y_joint",
    "RHand_R1Z_joint", "RHand_R1Y_joint", "RHand_R2Y_joint", "RHand_R3Y_joint",
    "RHand_T1Z_joint", "RHand_T1Y_joint", "RHand_T2Y_joint", "RHand_T3Y_joint",
]


class MujocoQposViz:
    def __init__(self, mjcf_path, launch_viewer=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        if len(JOINT_ORDER) != self.model.nq:
            raise ValueError(f"JOINT_ORDER has {len(JOINT_ORDER)} joints but model nq={self.model.nq}")
        self.qpos_addr = np.array([self.model.joint(name).qposadr[0] for name in JOINT_ORDER])
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if launch_viewer else None

    def set_qpos(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] != len(JOINT_ORDER):
            raise ValueError(f"expected {len(JOINT_ORDER)}-dim action, got {action.shape}")
        self.data.qpos[self.qpos_addr] = action
        mujoco.mj_forward(self.model, self.data)  # kinematics only -- no mj_step, no ctrl

    def sync(self):
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
