import sys
from pathlib import Path

# Reuse publisher helpers and app lifecycle from the original script.
# Importing this module instantiates SimulationApp first (required by Isaac Sim).
import ros2_depth as base

import carb
import numpy as np
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import nucleus, stage
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

BACKGROUND_STAGE_PATH = "/background"
LOCAL_BACKGROUND_USD = Path(__file__).resolve().parent / "test.usda"
FALLBACK_BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"


def resolve_background_usd() -> str:
    if LOCAL_BACKGROUND_USD.exists():
        # Isaac Sim can load local files via file:// URLs.
        return LOCAL_BACKGROUND_USD.resolve().as_uri()

    assets_root_path = nucleus.get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find local robot_table.usd or Isaac Sim assets folder")
        base.simulation_app.close()
        sys.exit()

    return assets_root_path + FALLBACK_BACKGROUND_USD_PATH


def run_scene():
    simulation_context = SimulationContext(stage_units_in_meters=1.0)

    # background_usd = resolve_background_usd()
    # carb.log_info(f"Loading scene: {background_usd}")
    # stage.add_reference_to_stage(background_usd, BACKGROUND_STAGE_PATH)

    camera = Camera(
        prim_path="/World/floating_camera",
        position=np.array([-3.11, -1.87, 1.0]),
        frequency=20,
        resolution=(256, 256),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
    )
    camera.initialize()

    base.simulation_app.update()
    camera.initialize()

    approx_freq = 30
    base.publish_camera_tf(camera)
    base.publish_camera_info(camera, approx_freq)
    base.publish_rgb(camera, approx_freq)
    base.publish_depth(camera, approx_freq)
    base.publish_pointcloud_from_depth(camera, approx_freq)

    simulation_context.initialize_physics()
    simulation_context.play()

    while base.simulation_app.is_running():
        simulation_context.step(render=True)

    simulation_context.stop()
    base.simulation_app.close()


if __name__ == "__main__":
    run_scene()
