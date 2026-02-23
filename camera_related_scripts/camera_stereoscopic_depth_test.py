# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument(
    "--scene_usd",
    type=str,
    default="/home/mingqian/Desktop/IsaacSim_test/pallet.usd",
    help="Path to a USD file to reference into the scene",
)
parser.add_argument(
    "--target_center",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    metavar=("X", "Y", "Z"),
    help="World-space center of the target object",
)
parser.add_argument(
    "--camera_height",
    type=float,
    default=2.0,
    help="Height offset above target_center for a top-down view",
)
parser.add_argument(
    "--use_scene_bbox_center",
    default=False,
    action="store_true",
    help="Use /World/custom_scene bbox center as camera target center",
)
parser.add_argument(
    "--max_distance",
    type=float,
    default=20.0,
    help="Depth clipping max distance in meters",
)
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.test})

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.sensors.camera import SingleViewDepthSensor
from isaacsim.storage.native.nucleus import get_assets_root_path
from pxr import UsdGeom

# Create a world
world = World(stage_units_in_meters=1.0)


def get_prim_world_bbox_center(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedBox()
    if aligned_box.IsEmpty():
        return None
    min_pt = aligned_box.GetMin()
    max_pt = aligned_box.GetMax()
    return np.array(
        [
            0.5 * (min_pt[0] + max_pt[0]),
            0.5 * (min_pt[1] + max_pt[1]),
            0.5 * (min_pt[2] + max_pt[2]),
        ],
        dtype=float,
    )

# Reference a saved USD scene into the current stage
omni.kit.commands.execute(
    "CreateReferenceCommand",
    usd_context=omni.usd.get_context(),
    path_to="/World/custom_scene",
    asset_path=args.scene_usd,
    instanceable=False,
)

# Place the camera directly above the target center and point downward.
target_center = np.array(args.target_center, dtype=float)
if args.use_scene_bbox_center:
    bbox_center = get_prim_world_bbox_center("/World/custom_scene")
    if bbox_center is not None:
        target_center = bbox_center
    else:
        print("Warning: failed to get /World/custom_scene bbox center, fallback to --target_center")
camera_position = np.array([target_center[0], target_center[1], target_center[2] + args.camera_height], dtype=float)
camera_orientation = rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True)

# Add a stereoscopic camera
camera = SingleViewDepthSensor(
    prim_path="/World/camera",
    name="depth_camera",
    position=camera_position,
    orientation=camera_orientation,
    frequency=20,
    resolution=(1920, 1080),
)

# Initialize the black grid scene for the background
assets_root_path = get_assets_root_path()
path_to = omni.kit.commands.execute(
    "CreateReferenceCommand",
    usd_context=omni.usd.get_context(),
    path_to="/World/black_grid",
    asset_path=assets_root_path + "/Isaac/Environments/Grid/gridroom_black.usd",
    instanceable=False,
)

# Reset the world state
world.reset()

# Initialize the camera, applying the appropriate schemas to the render product to enable depth sensing
camera.initialize(attach_rgb_annotator=False)

# Now that the camera is initialized, we can configure its parameters
# First, camera lens parameters
camera.set_focal_length(1.814756)
camera.set_focus_distance(400.0)
# Next, depth sensor parameters
camera.set_baseline_mm(55)
camera.set_focal_length_pixel(891.0)
camera.set_sensor_size_pixel(1280.0)
camera.set_max_disparity_pixel(110.0)
camera.set_confidence_threshold(0.99)
camera.set_noise_mean(0.5)
camera.set_noise_sigma(1.0)
camera.set_noise_downscale_factor_pixel(1.0)
camera.set_min_distance(0.5)
camera.set_max_distance(args.max_distance)

# Attach the DepthSensorDistance annotator and DistanceToImagePlane annotator to the camera
camera.attach_annotator("DepthSensorDistance")
camera.attach_annotator("distance_to_image_plane")

# Run for 10 frames in test mode
i = 0
while simulation_app.is_running() and (not args.test or i < 10):
    world.step(render=True)
    i += 1

# Saved the rendered frames as PNGs
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.test.utils")
from isaacsim.test.utils import save_depth_image

latest_frame = camera.get_current_frame()
save_depth_image(latest_frame["DepthSensorDistance"], "testing", "depth_sensor_distance.png", normalize=True)
save_depth_image(latest_frame["distance_to_image_plane"], "testing", "distance_to_image_plane.png", normalize=True)

simulation_app.close()
