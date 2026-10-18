"""Robot-only FK and local differential IK from the versioned SO-101 MJCF.

No simulator, mesh loading, global IK, object truth, or motor access. Positions are
meters in the model root, rotations wxyz, input arm joints degrees, gripper percent.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from liminal.core.types import SO101_JOINTS


def finite_vector(value, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"expected {length} numeric values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("non-finite or nonnumeric vector")
    return [float(v) for v in value]


class SO101Kinematics:
    def __init__(self, model: str, signs: list[float], offsets_deg: list[float], site: str = "gripperframe"):
        import numpy as np

        self.np = np
        self.signs = np.array(finite_vector(signs, 5))
        self.offsets = np.array(finite_vector(offsets_deg, 5))
        if not all(abs(s) == 1 for s in self.signs):
            raise ValueError("kinematic joint signs must be +1 or -1")
        root = ET.parse(Path(model)).getroot()
        compiler = root.find("compiler")
        if compiler is None or compiler.get("angle") != "radian":
            raise ValueError("kinematic MJCF must use radians")
        self.chain = []

        def find(body, parents):
            chain = parents + [body]
            target = body.find(f"site[@name='{site}']")
            if target is not None:
                return chain, target
            for child in body.findall("body"):
                result = find(child, chain)
                if result:
                    return result
            return None

        result = None
        for body in root.findall("worldbody/body"):
            result = find(body, [])
            if result:
                break
        if not result:
            raise ValueError("end-effector site missing from kinematic model")
        bodies, target = result
        names = []
        for body in bodies:
            origin = self._transform(body)
            joints = body.findall("joint")
            if len(joints) > 1:
                raise ValueError("expected one hinge per SO-101 body")
            joint = joints[0] if joints else None
            if joint is None:
                self.chain.append((origin, None, None, None))
            else:
                name = joint.get("name")
                if name not in SO101_JOINTS[:5] or joint.get("type", "hinge") != "hinge":
                    raise ValueError("unsupported kinematic joint")
                names.append(name)
                axis = np.array([float(x) for x in joint.get("axis", "0 0 1").split()])
                axis /= np.linalg.norm(axis)
                pos = np.array([float(x) for x in joint.get("pos", "0 0 0").split()])
                if float(joint.get("ref", "0")) != 0:
                    raise ValueError("nonzero joint reference unsupported")
                self.chain.append((origin, SO101_JOINTS.index(name), axis, pos))
        if tuple(names) != SO101_JOINTS[:5]:
            raise ValueError("kinematic chain does not match the SO-101 joint order")
        self.tip = self._transform(target)

    def rotation(self, quaternion):
        np = self.np
        q = np.array(finite_vector(quaternion, 4))
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            raise ValueError("zero quaternion")
        w, x, y, z = q / norm
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def rotvec(self, rotation):
        np = self.np
        # Quaternion extraction remains stable at rotations near pi.
        q = self.quaternion(rotation)
        if q[0] < 0:
            q = [-v for v in q]
        v = np.array(q[1:])
        n = np.linalg.norm(v)
        return 2*v if n < 1e-10 else v * (2*math.atan2(n, q[0])/n)

    def quaternion(self, r):
        np = self.np
        # Symmetric eigenproblem, component order x,y,z,w.
        k = np.array([[r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
                      [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
                      [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                      [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], r.trace()]]) / 3
        _, vecs = np.linalg.eigh(k)
        x, y, z, w = vecs[:, -1]
        q = [float(w), float(x), float(y), float(z)]
        return q if q[0] >= 0 else [-v for v in q]

    def exp(self, v):
        np = self.np
        angle = np.linalg.norm(v)
        if angle < 1e-12:
            return np.eye(3)
        x, y, z = np.asarray(v)/angle
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)

    def _transform(self, element):
        np = self.np
        if any(key in element.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("only quaternion MJCF orientations supported")
        result = np.eye(4)
        result[:3, 3] = [float(v) for v in element.get("pos", "0 0 0").split()]
        result[:3, :3] = self.rotation([float(v) for v in element.get("quat", "1 0 0 0").split()])
        return result

    def matrix(self, joints):
        np = self.np
        joints = finite_vector(joints, 6)
        radians = np.deg2rad((np.array(joints[:5])-self.offsets)/self.signs)
        transform = np.eye(4)
        for origin, index, axis, pos in self.chain:
            transform = transform @ origin
            if index is not None:
                turn = np.eye(4)
                turn[:3, :3] = self.exp(axis*radians[index])
                turn[:3, 3] = pos - turn[:3, :3] @ pos
                transform = transform @ turn
        return transform @ self.tip

    def forward(self, joints):
        t = self.matrix(joints)
        return {"position": t[:3, 3].tolist(), "quaternion_wxyz": self.quaternion(t[:3, :3]),
                "gripper_percent": finite_vector(joints, 6)[5]}

    def step(self, current, target, lower, upper, *, max_joint_step_deg=2.8647889757,
             max_gripper_step=10.0):
        """One damped least-squares update, recomputed after each actual ACK.

        Five joints cannot realize arbitrary six-dimensional poses. This is a local
        bounded update, never a claim of target arrival or task completion.
        """
        np = self.np
        current = finite_vector(current, 6)
        lower, upper = finite_vector(lower, 6), finite_vector(upper, 6)
        if any(not lo <= q <= hi for q, lo, hi in zip(current, lower, upper, strict=True)):
            raise ValueError("measured joints outside calibrated travel")
        pose = self.matrix(current)
        dp = np.array(target["position"]) - pose[:3, 3]
        dr = self.rotvec(self.rotation(target["quaternion_wxyz"]) @ pose[:3, :3].T)
        error = np.r_[dp * min(1, .02/max(np.linalg.norm(dp), 1e-12)),
                         dr * min(1, .1/max(np.linalg.norm(dr), 1e-12))]
        jacobian = np.zeros((6, 5))
        eps = 1e-5
        for j in range(5):
            shifted = list(current)
            shifted[j] += math.degrees(eps)
            t = self.matrix(shifted)
            jacobian[:3, j] = (t[:3, 3]-pose[:3, 3])/eps
            jacobian[3:, j] = self.rotvec(t[:3, :3] @ pose[:3, :3].T)/eps
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .05**2*np.eye(6), error)
        delta = np.clip(np.rad2deg(dq), -max_joint_step_deg, max_joint_step_deg)
        result = np.r_[np.array(current[:5])+delta,
                       current[5]+np.clip(target["gripper_percent"]-current[5], -max_gripper_step, max_gripper_step)]
        result = np.clip(result, lower, upper)
        return finite_vector(result.tolist(), 6)
