"""Read-only health check of one SO-101 follower bus: IDs, baud rate, supply voltage against each servo's limits.

Never writes a register and never enables torque. Run it after power-on and before calibration or any motion:

    python ws4/single_arm/probe_bus.py --port /dev/ttyACM0

Exit code 0 only when the six SO-101 IDs answer at 1 Mbps and every servo's measured voltage lies inside its own
Min/Max_Voltage_Limit (STS3215 units of 0.1 V, lerobot 0.5.1 motors/feetech/tables.py:53-54, 86).
"""

from __future__ import annotations

import argparse
import sys

SO101_IDS = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
SO101_BAUD = 1_000_000  # SOFollower bus default (lerobot motors/feetech/feetech.py DEFAULT_BAUDRATE)
REGISTERS = (
    "Model_Number",
    "Min_Voltage_Limit",
    "Max_Voltage_Limit",
    "Present_Voltage",
    "Present_Temperature",
    "Max_Temperature_Limit",
    "Torque_Enable",
    "Present_Position",
)


def probe(port: str) -> int:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    print(f"scanning {port} at every Feetech baud rate (broadcast ping, read-only) ...")
    found = FeetechMotorsBus.scan_port(port)
    for baud, ids in found.items():
        print(f"  {baud} bps: ids {ids}")
    ids = found.get(SO101_BAUD, [])
    ok = sorted(ids) == sorted(SO101_IDS)
    if not ok:
        print(f"FAIL: expected ids {sorted(SO101_IDS)} at {SO101_BAUD}; run lerobot-setup-motors if they differ")
        if not ids:
            return 2

    motors = {SO101_IDS.get(i, f"id{i}"): Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in sorted(ids)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)
    try:
        print(f"{'motor':14s} " + " ".join(f"{r:>21s}" for r in REGISTERS))
        for name in motors:
            values = {r: bus.read(r, name, normalize=False) for r in REGISTERS}
            print(f"{name:14s} " + " ".join(f"{values[r]:>21}" for r in REGISTERS))
            v, vmin, vmax = values["Present_Voltage"], values["Min_Voltage_Limit"], values["Max_Voltage_Limit"]
            if not vmin <= v <= vmax:
                print(f"FAIL: {name} supply {v / 10:.1f} V outside its limits {vmin / 10:.1f}..{vmax / 10:.1f} V")
                ok = False
            if values["Present_Temperature"] >= values["Max_Temperature_Limit"] - 10:
                print(f"FAIL: {name} is at {values['Present_Temperature']} C")
                ok = False
        volts = [bus.read("Present_Voltage", n, normalize=False) / 10 for n in motors]
        print(f"supply voltage: {min(volts):.1f} to {max(volts):.1f} V")
    finally:
        bus.port_handler.closePort()  # plain close: disconnect() would try to write Torque_Enable
    print("OK" if ok else "NOT OK")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", default="/dev/ttyACM0")
    sys.exit(probe(parser.parse_args().port))


if __name__ == "__main__":
    main()
