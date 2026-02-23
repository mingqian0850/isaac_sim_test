import argparse
import os
import sys

import carb
from isaacsim import SimulationApp

BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish semantic pointcloud from Isaac Sim to ROS2."
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode.")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to publish. 0 means run forever.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Warmup frames before export.")
    parser.add_argument(
        "--topic-name",
        type=str,
        default="/floating_camera_semantic_pointcloud",
        help="ROS2 topic name for semantic pointcloud.",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Optionally save each frame to npz while publishing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.getcwd(), "semantic_pointcloud_output"),
        help="Directory used when --save-npz is enabled.",
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": args.headless})

import numpy as np
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, nucleus, stage
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.nodes.scripts.utils import set_target_prims
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils


extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


def extract_pointcloud_payload(payload):
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected pointcloud payload type: {type(payload)}")

    points = payload.get("data")
    info = payload.get("info", {})
    semantic_ids = info.get("pointSemantic")
    instance_ids = info.get("pointInstance")

    if points is None:
        raise RuntimeError("Pointcloud payload does not contain 'data'.")

    points = np.asarray(points)
    semantic_ids = np.asarray(semantic_ids) if semantic_ids is not None else np.array([], dtype=np.int32)
    instance_ids = np.asarray(instance_ids) if instance_ids is not None else np.array([], dtype=np.int32)
    return points, semantic_ids, instance_ids


class SemanticPointCloudRos2Publisher:
    def __init__(self, topic_name: str, frame_id: str):
        try:
            import rclpy
            from rclpy.qos import QoSProfile
            from sensor_msgs.msg import PointCloud2, PointField
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rclpy/sensor_msgs not found. Source ROS2 setup.bash before launching Isaac Sim."
            ) from exc

        self._rclpy = rclpy
        self._PointCloud2 = PointCloud2
        self._PointField = PointField

        if not self._rclpy.ok():
            self._rclpy.init(args=None)
        self._node = self._rclpy.create_node("isaac_semantic_pointcloud_publisher")
        self._publisher = self._node.create_publisher(PointCloud2, topic_name, QoSProfile(depth=1))
        self._frame_id = frame_id

    @staticmethod
    def _semantic_to_rgb_float(semantic_ids: np.ndarray) -> np.ndarray:
        # Create deterministic colors from semantic ids for RViz RGB mode.
        sid = semantic_ids.astype(np.uint32, copy=False)
        r = (sid * 1664525 + 1013904223) & np.uint32(0xFF)
        g = (sid * 22695477 + 1) & np.uint32(0xFF)
        b = (sid * 1103515245 + 12345) & np.uint32(0xFF)
        rgb_uint32 = (r << np.uint32(16)) | (g << np.uint32(8)) | b
        return rgb_uint32.view(np.float32)

    def publish(self, points_xyz: np.ndarray, semantic_ids: np.ndarray):
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise RuntimeError(f"Expected pointcloud shape (N, 3), got {points_xyz.shape}")
        if semantic_ids.size != points_xyz.shape[0]:
            semantic_ids = np.zeros(points_xyz.shape[0], dtype=np.int32)

        cloud = np.zeros(
            points_xyz.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("semantic_id", np.uint32),
                ("rgb", np.float32),
            ],
        )
        cloud["x"] = points_xyz[:, 0]
        cloud["y"] = points_xyz[:, 1]
        cloud["z"] = points_xyz[:, 2]
        cloud["semantic_id"] = semantic_ids.astype(np.uint32, copy=False)
        cloud["rgb"] = self._semantic_to_rgb_float(semantic_ids)

        msg = self._PointCloud2()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = cloud.shape[0]
        msg.fields = [
            self._PointField(name="x", offset=0, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="y", offset=4, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="z", offset=8, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="semantic_id", offset=12, datatype=self._PointField.UINT32, count=1),
            self._PointField(name="rgb", offset=16, datatype=self._PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = (sys.byteorder != "little")
        msg.point_step = 20
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = cloud.tobytes()
        self._publisher.publish(msg)
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self):
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def publish_camera_info(camera: Camera, freq: int):
    from isaacsim.ros2.bridge import read_camera_info

    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_camera_info"
    frame_id = camera.prim_path.split("/")[-1]

    writer = rep.writers.get("ROS2PublishCameraInfo")
    camera_info, _ = read_camera_info(render_product_path=render_product)
    writer.initialize(
        frameId=frame_id,
        nodeNamespace="",
        queueSize=1,
        topicName=topic_name,
        width=camera_info.width,
        height=camera_info.height,
        projectionType=camera_info.distortion_model,
        k=camera_info.k.reshape([1, 9]),
        r=camera_info.r.reshape([1, 9]),
        p=camera_info.p.reshape([1, 12]),
        physicalDistortionModel=camera_info.distortion_model,
        physicalDistortionCoefficients=camera_info.d,
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_rgb(camera: Camera, freq: int):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_rgb"
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(frameId=frame_id, nodeNamespace="", queueSize=1, topicName=topic_name)
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_depth(camera: Camera, freq: int):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_depth"
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.DistanceToImagePlane.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(frameId=frame_id, nodeNamespace="", queueSize=1, topicName=topic_name)
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_pointcloud_from_depth(camera: Camera, freq: int):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_pointcloud"
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.DistanceToImagePlane.name)
    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(frameId=frame_id, nodeNamespace="", queueSize=1, topicName=topic_name)
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_camera_tf(camera: Camera):
    camera_prim = camera.prim_path
    if not is_prim_path_valid(camera_prim):
        raise ValueError(f"Camera path '{camera_prim}' is invalid.")

    camera_frame_id = camera_prim.split("/")[-1]
    ros_camera_graph_path = "/CameraTFActionGraph"

    if not is_prim_path_valid(ros_camera_graph_path):
        og.Controller.edit(
            {
                "graph_path": ros_camera_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("IsaacClock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("RosPublisher", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "RosPublisher.inputs:execIn"),
                    ("IsaacClock.outputs:simulationTime", "RosPublisher.inputs:timeStamp"),
                ],
            },
        )

    og.Controller.edit(
        ros_camera_graph_path,
        {
            og.Controller.Keys.CREATE_NODES: [
                ("PublishTF_" + camera_frame_id, "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("PublishRawTF_" + camera_frame_id + "_world", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("PublishRawTF_world_to_World", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishTF_" + camera_frame_id + ".inputs:topicName", "/tf"),
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:topicName", "/tf"),
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:parentFrameId", camera_frame_id),
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:childFrameId", camera_frame_id + "_world"),
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:rotation", [0.5, -0.5, 0.5, 0.5]),
                ("PublishRawTF_world_to_World.inputs:topicName", "/tf"),
                ("PublishRawTF_world_to_World.inputs:parentFrameId", "world"),
                ("PublishRawTF_world_to_World.inputs:childFrameId", "World"),
                ("PublishRawTF_world_to_World.inputs:translation", [0.0, 0.0, 0.0]),
                ("PublishRawTF_world_to_World.inputs:rotation", [0.0, 0.0, 0.0, 1.0]),
            ],
            og.Controller.Keys.CONNECT: [
                (ros_camera_graph_path + "/OnTick.outputs:tick", "PublishTF_" + camera_frame_id + ".inputs:execIn"),
                (
                    ros_camera_graph_path + "/OnTick.outputs:tick",
                    "PublishRawTF_" + camera_frame_id + "_world.inputs:execIn",
                ),
                (ros_camera_graph_path + "/OnTick.outputs:tick", "PublishRawTF_world_to_World.inputs:execIn"),
                (
                    ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                    "PublishTF_" + camera_frame_id + ".inputs:timeStamp",
                ),
                (
                    ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                    "PublishRawTF_" + camera_frame_id + "_world.inputs:timeStamp",
                ),
                (
                    ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                    "PublishRawTF_world_to_World.inputs:timeStamp",
                ),
            ],
        },
    )

    set_target_prims(
        primPath=ros_camera_graph_path + "/PublishTF_" + camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim],
    )


def run_scene():
    if args.save_npz:
        os.makedirs(args.output_dir, exist_ok=True)
    simulation_context = SimulationContext(stage_units_in_meters=1.0)

    assets_root_path = nucleus.get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder.")
        simulation_app.close()
        sys.exit(1)

    stage.add_reference_to_stage(assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH)

    camera = Camera(
        prim_path="/World/floating_camera",
        position=np.array([-3.11, -1.87, 1.0]),
        frequency=20,
        resolution=(256, 256),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
    )
    camera.initialize()
    simulation_app.update()
    camera.initialize()

    approx_freq = 30
    publish_camera_tf(camera)
    publish_camera_info(camera, approx_freq)
    publish_rgb(camera, approx_freq)
    publish_depth(camera, approx_freq)
    publish_pointcloud_from_depth(camera, approx_freq)

    pointcloud_annotator = rep.annotators.get("pointcloud")
    pointcloud_annotator.attach([camera._render_product_path])
    camera_frame_id = camera.prim_path.split("/")[-1]
    # Replicator pointcloud annotator outputs points in world coordinates.
    # Publish with World frame_id to avoid an extra incorrect TF transform in RViz.
    semantic_frame_id = "World"
    ros2_pub = SemanticPointCloudRos2Publisher(topic_name=args.topic_name, frame_id=semantic_frame_id)

    simulation_context.initialize_physics()
    simulation_context.play()

    for _ in range(args.warmup_frames):
        simulation_context.step(render=True)

    frame_idx = 0
    while simulation_app.is_running():
        if args.frames > 0 and frame_idx >= args.frames:
            break
        simulation_context.step(render=True)

        payload = pointcloud_annotator.get_data()
        points_xyz, semantic_ids, instance_ids = extract_pointcloud_payload(payload)
        ros2_pub.publish(points_xyz=points_xyz, semantic_ids=semantic_ids)

        if args.save_npz:
            output_file = os.path.join(args.output_dir, f"frame_{frame_idx:06d}.npz")
            np.savez_compressed(
                output_file,
                xyz=points_xyz.astype(np.float32, copy=False),
                semantic_id=semantic_ids.astype(np.int32, copy=False),
                instance_id=instance_ids.astype(np.int32, copy=False),
            )

        frame_idx += 1

    simulation_context.stop()
    ros2_pub.close()
    simulation_app.close()
    print(f"Published semantic pointcloud to ROS2 topic: {args.topic_name} (frame_id={semantic_frame_id})")
    if args.save_npz:
        print(f"Saved semantic pointcloud files to: {args.output_dir}")


if __name__ == "__main__":
    try:
        run_scene()
    except Exception as exc:
        carb.log_error(f"Failed to export semantic pointcloud: {exc}")
        simulation_app.close()
        raise
