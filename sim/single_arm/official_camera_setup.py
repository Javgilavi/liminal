"""SO-101 camera layout checked against BFL's published preflight image.

The BFL guide publishes roles and an example image, not calibrated extrinsics.
Scene placement is an approximation; wrist placement is the Menagerie lens pose.
This module changes cameras only, without hiding robot geometry or changing policy.
"""

from __future__ import annotations

SCENE_EYE = (0.55, -0.30, 0.65)
SCENE_TARGET = (0.16, 0.03, 0.045)
SCENE_FOVY = 40.0
WRIST_POSITION = (-0.00296, 0.05436, -0.05310)
WRIST_QUATERNION = (0.958280, -0.285792, -0.001556, 0.004547)
WRIST_FOVY = 61.0
LENS_CLEARANCE_M = 0.001


def configure(model, data):
    import mujoco
    import numpy as np

    eye = np.asarray(SCENE_EYE)
    z = eye - SCENE_TARGET
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    scene = model.camera("scene").id
    model.cam_pos[scene] = eye
    model.cam_quat[scene] = quat
    model.cam_fovy[scene] = SCENE_FOVY
    wrist = model.camera("wrist_cam").id
    model.cam_pos[wrist] = WRIST_POSITION
    model.cam_quat[wrist] = np.asarray(WRIST_QUATERNION) / np.linalg.norm(WRIST_QUATERNION)
    # The stock optical center is ~0.5 mm inside the visual lens cylinder.
    # Place it 1 mm forward to see out of that lens, rather than hide any mesh.
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.cam_quat[wrist])
    model.cam_pos[wrist] += -rotation.reshape(3, 3)[:, 2] * LENS_CLEARANCE_M
    model.cam_fovy[wrist] = WRIST_FOVY
    mujoco.mj_forward(model, data)


def description():
    return {
        "official_reference": "https://github.com/black-forest-labs/flux-action/blob/main/docs/so101-lora.md#so-101-camera-keys-and-tiling",
        "reference_image": "https://raw.githubusercontent.com/black-forest-labs/flux-action/main/docs/assets/so101_camera_tiling.png",
        "scene": {
            "position_m": SCENE_EYE,
            "target_m": SCENE_TARGET,
            "fovy_deg": SCENE_FOVY,
            "calibration": ("VERIFY: visually matched overhead-oblique view; "
                            "BFL does not publish exact intrinsics/extrinsics"),
        },
        "wrist": {
            "parent_body": "camera_mount",
            "position_m": WRIST_POSITION,
            "quaternion_wxyz": WRIST_QUATERNION,
            "fovy_deg": WRIST_FOVY,
            "optical_center_forward_clearance_m": LENS_CLEARANCE_M,
            "calibration": "Menagerie SO-101 physical lens pose; not a measured BFL camera calibration",
        },
        "input_order": ["images.scene", "images.wrist"],
        "canvas_hw": [256, 512],
        "geometry_hidden": False,
        "policy_or_normalization_changes": False,
    }
