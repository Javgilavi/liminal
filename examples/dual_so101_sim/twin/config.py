"""Measured or estimated dimensions of the real Haller tabletop (metres, kilograms).

Every value marked EST comes from `erh_hack/WhatsApp Video 2026-10-18 at 01.08.45.mp4` and typical
object sizes; replace with ruler measurements when available. World frame is the one of
`../scene.xml`: bench/table top at z = 0, arms on a 0.10 m riser at x = ±0.15, reaching toward −y.
"""

TABLE = dict(radius=0.40, center=(0.0, -0.18))  # round white café table, EST
MAT = dict(size=(0.30, 0.24, 0.012), center=(0.20, -0.22))  # black padded laptop sleeve, EST

BOX = dict(  # white cardboard box, lid hinged on the side facing the robot (+y), EST
    center=(-0.07, -0.21), outer=(0.26, 0.18, 0.055), wall=0.004, lid_thickness=0.003,
    mass=0.12, lid_mass=0.04, hinge_damping=0.02, hinge_frictionloss=0.02,  # taped hinge holds the lid where left
    lid_open_deg=110.0,  # fully open: lid leans back toward the robot
    lid_overhang=0.025,  # lid extends past the front wall; its flap hangs in front with a finger gap, EST
)

MARKER = dict(radius=0.0085, length=0.135, cap_length=0.045, cap_radius=0.0095, mass=0.016)  # EST
MARKERS = {
    "green_marker": dict(body_rgba=(0.10, 0.42, 0.25, 1), cap_rgba=(0.10, 0.42, 0.25, 1)),
    "red_marker": dict(body_rgba=(0.82, 0.12, 0.08, 1), cap_rgba=(0.08, 0.08, 0.08, 1)),
}
CUTTER = dict(size=(0.155, 0.026, 0.018), mass=0.045, rgba=(0.98, 0.76, 0.05, 1))  # yellow box cutter, EST

ARM_RGBA = (0.10, 0.46, 0.36, 1)  # green 3D-printed parts in the video
TOWER_RGBA = (0.10, 0.46, 0.36, 1)
TABLE_RGBA = (0.93, 0.93, 0.91, 1)
MAT_RGBA = (0.05, 0.05, 0.06, 1)
BOX_RGBA = (0.95, 0.95, 0.93, 1)
TAPE_RGBA = (0.55, 0.38, 0.24, 1)

# Policy cameras (match the real robot): D455 colour on the tower and the active arm's wrist camera.
SCENE_CAMERA = "top"
SCENE_WH = (640, 480)
WRIST_WH = (640, 360)
