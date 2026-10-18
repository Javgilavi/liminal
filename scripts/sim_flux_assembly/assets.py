"""Build documented NIST CAD visuals and hollow compound collision geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_flux_lora"))
from common import SIM_ROOT  # noqa: E402


def triangles_from_stl(blob):
    import numpy as np

    count = struct.unpack("<I", blob[80:84])[0]
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(blob, dtype=dtype, offset=84, count=count)["vertices"].astype(float)


def write_stl(path, triangles):
    import numpy as np

    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.zeros(len(triangles), dtype=dtype)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-20)
    records["normal"], records["vertices"] = normals, triangles
    path.write_bytes(
        b"NIST CAD-derived SO101 assembly".ljust(80, b"\0") + struct.pack("<I", len(records)) + records.tobytes()
    )


def hollow_layer(asset, body, name, inner, outer, bottom, top, mass, sectors=12):
    import numpy as np

    for i in range(sectors):
        vertices = [
            [radius * np.cos(angle), radius * np.sin(angle), z]
            for z in [bottom, top]
            for radius in [inner, outer]
            for angle in [2 * np.pi * i / sectors, 2 * np.pi * (i + 1) / sectors]
        ]
        ET.SubElement(asset, "mesh", name=f"{name}_{i}", vertex=" ".join(str(v) for point in vertices for v in point))
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_{i}",
            type="mesh",
            mesh=f"{name}_{i}",
            group="3",
            mass=str(mass / sectors),
            rgba="0.2 0.8 0.2 0.2",
            friction="1 0.01 0.001",
            condim="3",
            solref="0.01 1",
        )


def build(root: Path):
    import numpy as np

    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    archive = assets / "source/nist-stl.zip"
    parts = {
        "gear": ("Gear_20teeth", (1.6, 1.6, 1.6)),
        "collar": ("NCLC10-12-20.0-1", (1.6, 1.6, 1.6)),
        "shaft": ("SFAD10-31-F20-B16-P8-N6-1", (1.2, 1.2, 50 / 51)),
    }
    provenance = {}
    with zipfile.ZipFile(archive) as source:
        for name, (token, scale) in parts.items():
            entry = next(p for p in source.namelist() if token in p)
            blob = source.read(entry)
            triangles = triangles_from_stl(blob)
            points = triangles.reshape(-1, 3)
            lower, upper = points.min(0), points.max(0)
            center = (lower + upper) / 2
            # Source models are in millimetres, posed on the full task board;
            # their shaft axis is +Y. Recenter and rotate +Y onto world +Z.
            transformed = (
                np.stack(
                    [triangles[:, :, 0] - center[0], -(triangles[:, :, 2] - center[2]), triangles[:, :, 1] - lower[1]],
                    axis=-1,
                )
                * np.array(scale)
                / 1000
            )
            write_stl(assets / f"{name}.stl", transformed)
            provenance[name] = {
                "archive_entry": entry,
                "source_sha256": hashlib.sha256(blob).hexdigest(),
                "triangles": len(triangles),
                "scale_xyz_after_rotation": scale,
                "extent_m": np.ptp(transformed.reshape(-1, 3), axis=0).tolist(),
            }
    scene = ET.parse(SIM_ROOT / "scene.xml")
    mjcf = scene.getroot()
    mjcf.set("model", "so101_nist_gear_collar_assembly")
    mjcf.find("include").set("file", str(SIM_ROOT / "so101.xml"))
    world = mjcf.find("worldbody")
    for name in ["block", "bowl"]:
        world.remove(world.find(f"body[@name='{name}']"))
    asset = ET.SubElement(mjcf, "asset")
    for name in parts:
        ET.SubElement(asset, "mesh", name=f"{name}_visual", file=str(assets / f"{name}.stl"))
    for name, xy, color in [
        ("gear", (0.23, -0.045), "0.12 0.35 0.82 1"),
        ("collar", (0.23, -0.13), "0.78 0.46 0.12 1"),
    ]:
        body = ET.SubElement(world, "body", name=name, pos=f"{xy[0]} {xy[1]} 0.010")
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_visual",
            type="mesh",
            mesh=f"{name}_visual",
            mass="0",
            contype="0",
            conaffinity="0",
            group="2",
            rgba=color,
        )
        if name == "gear":
            hollow_layer(asset, body, "gear_teeth", 0.008, 0.0176, 0, 0.016, 0.020)
            hollow_layer(asset, body, "gear_hub", 0.008, 0.0128, 0.016, 0.032, 0.010)
        else:
            hollow_layer(asset, body, "collar_wall", 0.008, 0.0096, 0, 0.032, 0.020)
        tray = ET.SubElement(world, "body", name=f"{name}_tray", pos=f"{xy[0]} {xy[1]} 0")
        ET.SubElement(
            tray,
            "geom",
            name=f"{name}_support",
            type="box",
            size="0.034 0.032 0.007",
            pos="0 0 0.003",
            rgba="0.23 0.25 0.28 1",
            friction="1 0.01 0.001",
        )
    fixture = ET.SubElement(world, "body", name="fixture", pos="0.20 0.08 0")
    ET.SubElement(
        fixture,
        "geom",
        name="fixture_plate",
        type="box",
        size="0.052 0.041 0.007",
        pos="0 0 0.003",
        rgba="0.45 0.48 0.52 1",
        friction="0.6 0.01 0.001",
    )
    shaft = ET.SubElement(fixture, "body", name="shaft", pos="0 0 0.010")
    ET.SubElement(
        shaft,
        "geom",
        name="shaft_visual",
        type="mesh",
        mesh="shaft_visual",
        contype="0",
        conaffinity="0",
        group="2",
        rgba="0.65 0.68 0.72 1",
    )
    ET.SubElement(
        shaft,
        "geom",
        name="shaft_collision",
        type="mesh",
        mesh="shaft_visual",
        group="3",
        rgba="0.2 0.8 0.2 0.2",
        friction="0.3 0.005 0.001",
        solref="0.01 1",
    )
    for i, (x, y) in enumerate([(-0.04, -0.03), (0.04, -0.03), (-0.04, 0.03), (0.04, 0.03)]):
        ET.SubElement(
            fixture,
            "geom",
            name=f"fixture_screw_{i}",
            type="cylinder",
            size="0.003 0.002",
            pos=f"{x} {y} 0.012",
            rgba="0.12 0.13 0.14 1",
        )
    ET.indent(scene, "  ")
    scene.write(root / "scene.xml", encoding="unicode")
    manifest = {
        "source_page": "https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/robotic-grasping-0",
        "source_url": "https://www.nist.gov/system/files/documents/2022/02/16/STL%20Files_0.zip",
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "geometry": "NIST industrial CAD-derived meshes; not a scan or measured physical reconstruction",
        "parts": provenance,
        "collision": "12 convex annular sectors per layer preserve through-holes; tooth detail is visual only",
        "fit": "16 mm nominal bores over a shaft with 12 mm maximum diameter; teaching clearance 2 mm radial",
        "fixture": "custom fixed mounting plate and two part supports; no object welds or attachment constraints",
    }
    (assets / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build(parser.parse_args().root)
