"""Create a portable, colored GLB of the assembled CAD parts (NumPy only)."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def main(assets: Path, report_path: Path, output: Path):
    import numpy as np
    from assets import triangles_from_stl

    report = json.loads(report_path.read_text())
    center = np.array(report["metadata"]["fixture_xy"] + [0.0])
    spec = {
        "asset": {"version": "2.0", "generator": "Liminal NIST assembly CAD export"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "extras": {
            "source": "NIST ICRA 2022 practice task board CAD",
            "units": "metres",
            "reconstruction": "CAD-derived; scaled teaching geometry, not a physical scan",
        },
    }
    binary = bytearray()

    def append_accessor(values, target):
        values = np.asarray(values, dtype="<f4").reshape(-1, 3)
        offset = len(binary)
        binary.extend(values.tobytes())
        view = len(spec["bufferViews"])
        spec["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": values.nbytes, "target": target})
        index = len(spec["accessors"])
        spec["accessors"].append(
            {
                "bufferView": view,
                "componentType": 5126,
                "count": len(values),
                "type": "VEC3",
                "min": values.min(0).tolist(),
                "max": values.max(0).tolist(),
            }
        )
        return index

    for name, color in [
        ("gear", [0.12, 0.35, 0.82, 1]),
        ("collar", [0.78, 0.46, 0.12, 1]),
        ("shaft", [0.65, 0.68, 0.72, 1]),
        ("base", [0.45, 0.48, 0.52, 1]),
    ]:
        if name == "base":
            vertices = np.array([[x, y, z] for z in [-0.004, 0.010] for y in [-0.041, 0.041] for x in [-0.052, 0.052]])
            faces = [
                [0, 2, 1],
                [1, 2, 3],
                [4, 5, 6],
                [5, 7, 6],
                [0, 1, 4],
                [1, 5, 4],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [1, 3, 5],
                [3, 7, 5],
            ]
            triangles = vertices[faces]
            translation = np.zeros(3)
        else:
            triangles = triangles_from_stl((assets / f"{name}.stl").read_bytes())
            translation = (
                np.array(report["final_state"][name]["position"]) - center if name != "shaft" else [0, 0, 0.010]
            )
        # glTF uses Y up; convert MuJoCo's Z-up coordinates with a proper rotation.
        rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        triangles = triangles @ rotation.T
        translation = rotation @ translation
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
        positions = append_accessor(triangles.reshape(-1, 3), 34962)
        normal_index = append_accessor(np.repeat(normals, 3, axis=0), 34962)
        index = len(spec["meshes"])
        spec["materials"].append(
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": color,
                    "metallicFactor": 0.65 if name in ["shaft", "collar"] else 0.05,
                    "roughnessFactor": 0.35,
                },
            }
        )
        spec["meshes"].append(
            {
                "name": name,
                "primitives": [
                    {"attributes": {"POSITION": positions, "NORMAL": normal_index}, "material": index, "mode": 4}
                ],
            }
        )
        spec["nodes"].append({"name": name, "mesh": index, "translation": translation.tolist()})
        spec["scenes"][0]["nodes"].append(index)
    spec["buffers"] = [{"byteLength": len(binary)}]
    text = json.dumps(spec, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    size = 12 + 8 + len(text) + 8 + len(binary)
    blob = struct.pack("<III", 0x46546C67, 2, size)
    blob += struct.pack("<II", len(text), 0x4E4F534A) + text
    blob += struct.pack("<II", len(binary), 0x004E4942) + binary
    output.write_bytes(blob)
    print(json.dumps({"output": str(output), "bytes": len(blob), "objects": len(spec["nodes"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.assets, args.report, args.output)
