"""Central configuration. Tune behaviour here, not inside the modules."""
import math
import os

# --- Video input ---
CAMERA_INDEX = 0
FRAME_SIZE = (640, 480)
CAMERA_HFOV_DEG = 60.0              # Logitech C270 and most laptop webcams are ~60 deg

# --- Terrain (CLIP zero-shot on a grid of cells) ---
CLIP_MODEL = "openai/clip-vit-base-patch32"
GRID_COLS, GRID_ROWS = 10, 8
TERRAIN_EVERY_N_FRAMES = 3          # terrain changes slowly; reuse between frames
IMPASSABLE_MIN_CONF = 0.5           # only trust an impassable label (water/sky) above this
SKY_MAX_ROW_FRAC = 0.5              # sky can only appear in the top half of the frame

# label -> (CLIP prompt, traversal cost; math.inf = impassable)
TERRAIN = {
    "road":   ("a photo of a paved road or concrete ground",  1.0),
    "dirt":   ("a photo of dry dirt ground",                  1.5),
    "grass":  ("a photo of short grass",                      2.0),
    "gravel": ("a photo of gravel",                           2.5),
    "sand":   ("a photo of loose sand",                       4.0),
    "mud":    ("a photo of wet muddy ground",                 8.0),
    "rubble": ("a photo of rubble and debris",                9.0),
    "water":  ("a photo of water: a puddle, a stream or shallow water over rocks", 15.0),
    "sky":    ("a photo of the sky",                          math.inf),
    "indoor": ("a photo of an indoor floor",                  1.0),
}

# --- Depth (monocular, metric) ---
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
MAX_DEPTH_M = 80.0                  # model range; far cliffs and ridges still reach the 3D map
ODOM_MAX_DEPTH_M = 30.0             # odometry only trusts depth nearer than this
MAP_MAX_DEPTH_M = 35.0              # beyond this, small ground-plane tilt errors swamp the heights
MAP_MIN_ELEV_M = -5.0               # nothing sits further than this below the fitted ground

# --- Output layout: camera feed | depth map on top (50/50), four stat panels underneath ---
PANEL_SIZE = (320, 240)             # each bottom panel
OUTPUT_SIZE = (2 * FRAME_SIZE[0], FRAME_SIZE[1] + PANEL_SIZE[1])                      # 1280x720
CLOUD_STRIDE = 3                    # sample every Nth pixel for the point cloud

# --- Ground-point tracking (TAPNet / BootsTAPIR) ---
TAPNET_CHECKPOINT = "checkpoints/bootstapir_checkpoint_v2.pt"

# --- Simulated drone telemetry (window 2) ---
DRONE_HOVER_RPM = 5200              # typical 5-inch quad hover RPM
DRONE_BATTERY_DRAIN_PER_S = 0.08    # % per second, for the simulated battery gauge
MIN_BOX_DEPTH_M = 0.3               # minimum cuboid depth so far objects still read as 3D

# --- Obstacles (open-vocabulary YOLO) ---
YOLO_WEIGHTS = "yolov8s-worldv2.pt"
YOLO_CONF = 0.15
OBSTACLE_CLASSES = ["rock", "boulder", "tree", "tree trunk", "log", "bush",
                    "fence", "pole", "wall", "car", "person"]
# Ignore wide boxes touching the bottom edge: that's the camera vehicle's own hood
EGO_BOTTOM_FRAC, EGO_MIN_WIDTH_FRAC = 0.95, 0.35
EGO_MIN_TOP_FRAC = 0.6             # ...and whose top edge is in the lower 40% (a hood is short)
MAX_BOX_WIDTH_FRAC = 0.9            # a box spanning the full width is the background, not an obstacle
MAX_BOX_AREA_FRAC = 0.8            # boxes covering almost the whole frame are scene-level false positives

# --- Planner ---
PLAN_FIRST_ROW = 3                  # rows above this are horizon/trees: not drivable, not planned on
OBSTACLE_INFLATION = 1              # extra cells blocked around each obstacle
OBSTACLE_FOOTPRINT_ROWS = 2         # an obstacle blocks only its bottom grid rows (ground contact)
OBSTACLE_NEAR_PENALTY = 5.0         # added cost for cells next to the inflated zone

# --- Nemotron on Nebius Token Factory (OpenAI-compatible API) ---
# Check what your key can call with:  python nebius_models.py
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
# tier -> (preferred id, fallback keyword sets matched against the live model list)
MODEL_TIERS = {
    "fast":   (os.environ.get("COPILOT_MODEL", "nvidia/nvidia-nemotron-3-nano-30b-a3b"),
               [("nano",), ("super",)]),                 # 1 Hz in-flight decisions: latency first
    "reason": (os.environ.get("REPORT_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
               [("ultra",), ("super",)]),                # mission planning + debrief: reasoning first
}
COPILOT_EVERY_S = 1.0               # one decision per second of flight
REPORT_MAX_STEPS = 120              # last N logged decisions go into the mission report
TAVILY_MAX_RESULTS = 4              # per preflight web search
PLANNER_MAX_STEPS = 6               # tool-call rounds for the preflight agent

# --- Safety guard: deterministic rules that can veto a Nemotron decision ---
GUARD_STOP_DIST_M = 2.0             # obstacle this close inside the cone -> HOLD
GUARD_CONE_DEG = 20.0               # half-angle of the forward cone
GUARD_MIN_ALT_M = 1.0               # below this while descending -> CLIMB
GUARD_DESCENT_MPS = -1.5


# --- Scene profiles: the depth model and distance limits depend on where the camera is ---
SCENES = {
    "outdoor": dict(DEPTH_MODEL="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
                    MAX_DEPTH_M=80.0, ODOM_MAX_DEPTH_M=30.0, MAP_MAX_DEPTH_M=35.0,
                    PLANE_CONSTRAINED=False),   # FPV drones bank hard: no "stays level" prior
    # the outdoor model flattens indoor floors (no depth gradient), which breaks the ground plane
    "indoor": dict(DEPTH_MODEL="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                   MAX_DEPTH_M=20.0, ODOM_MAX_DEPTH_M=10.0, MAP_MAX_DEPTH_M=12.0,
                   PLANE_CONSTRAINED=True),    # hand-held: floor stays near level, walls must not win
}


PLANE_CONSTRAINED = False


def use_scene(name: str):
    """Switch profile. Call before building the Pipeline."""
    globals().update(SCENES[name])


def focal_px(width: int) -> float:
    """Pinhole focal length in pixels from the horizontal field of view."""
    return width / (2 * math.tan(math.radians(CAMERA_HFOV_DEG) / 2))


# --- Monad: on-chain flight recorder (contracts/FlightRecorder.sol) ---
# Chain 143 = Monad mainnet, 10143 = Monad testnet. Pick with MONAD_NETWORK in .env.
MONAD_NETWORKS = {
    "mainnet": ("https://rpc.monad.xyz", "https://monadvision.com"),
    "testnet": ("https://testnet-rpc.monad.xyz", "https://testnet.monadvision.com"),
}
MONAD_NETWORK = os.environ.get("MONAD_NETWORK", "testnet")
MONAD_RPC_URL = os.environ.get("MONAD_RPC_URL", MONAD_NETWORKS[MONAD_NETWORK][0])
MONAD_EXPLORER = os.environ.get("MONAD_EXPLORER", MONAD_NETWORKS[MONAD_NETWORK][1])
