"""Small numpy SE(3)/SO(3) helpers shared by the kinematics, EE50 codec and IK.

Everything is batched over leading axes where that is cheap. Rotations are 3x3 matrices, poses 4x4.
"""

from __future__ import annotations

import numpy as np


def rot_x(a: float | np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def rot_y(a: float | np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def rot_z(a: float | np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def make_pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def inv_pose(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make_pose(Rt, -np.einsum("...ij,...j->...i", Rt, p))


def axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues. ``axis`` (..., 3) unit vectors, ``angle`` (...)."""
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)[..., None, None]
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)], -2)
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v, axis=-1)
    safe = np.where(angle > 1e-12, angle, 1.0)
    axis = v / safe[..., None]
    axis = np.where((angle > 1e-12)[..., None], axis, np.array([1.0, 0.0, 0.0]))
    return axis_angle_matrix(axis, angle)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Log map, robust near 0 and pi."""
    R = np.asarray(R, dtype=float)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(angle)
    small = angle < 1e-6
    near_pi = angle > np.pi - 1e-4
    scale = np.where(small, 0.5, angle / np.where(small, 1.0, 2 * np.where(np.abs(s) > 1e-12, s, 1.0)))
    out = w * scale[..., None]
    if np.any(near_pi):
        # Near pi the antisymmetric part vanishes; recover the axis from the symmetric part.
        Rp = R[near_pi]
        B = (Rp + np.eye(3)) / 2  # = a a^T at exactly pi
        idx = np.argmax(np.diagonal(B, axis1=-2, axis2=-1), axis=-1)
        axis = B[np.arange(len(Rp)), :, idx]
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        wp = w[near_pi]
        sign = np.where(np.einsum("...i,...i->...", axis, wp) < 0, -1.0, 1.0)
        out[near_pi] = axis * (sign * angle[near_pi])[..., None]
    return out


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Geodesic angle of a rotation matrix (radians)."""
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    return np.arccos(cos)


def rotation_distance(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    return rotation_angle(np.swapaxes(Ra, -1, -2) @ Rb)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.einsum("...i,...i->...", a, b), -1.0, 1.0))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Closest rotation in Frobenius norm (SVD with det fix)."""
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(M.shape[:-2] + (3,))
    D[..., 2] = d
    return (U * D[..., None, :]) @ Vt


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
