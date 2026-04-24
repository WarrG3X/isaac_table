import argparse
import json
import os
from pathlib import Path

import numpy as np
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf.timestamp_pb2 import Timestamp
from mcap.writer import Writer

from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame


def build_file_descriptor_set(message_cls):
    seen = set()
    descriptor_set = FileDescriptorSet()

    def add_file(file_desc):
        if file_desc.name in seen:
            return
        for dependency in file_desc.dependencies:
            add_file(dependency)
        proto = descriptor_set.file.add()
        file_desc.CopyToProto(proto)
        seen.add(file_desc.name)

    add_file(message_cls.DESCRIPTOR.file)
    return descriptor_set.SerializeToString()


def make_timestamp(ts_seconds: float) -> Timestamp:
    stamp = Timestamp()
    whole = int(ts_seconds)
    nanos = int(round((ts_seconds - whole) * 1_000_000_000))
    if nanos >= 1_000_000_000:
        whole += 1
        nanos -= 1_000_000_000
    stamp.seconds = whole
    stamp.nanos = nanos
    return stamp


def register_json_schema(writer: Writer, topic: str, schema_name: str, schema_obj: dict) -> int:
    schema_id = writer.register_schema(
        name=schema_name,
        encoding="jsonschema",
        data=json.dumps(schema_obj, separators=(",", ":")).encode("utf-8"),
    )
    return writer.register_channel(topic=topic, message_encoding="json", schema_id=schema_id)


def register_protobuf_schema(writer: Writer, topic: str, message_cls) -> int:
    schema_id = writer.register_schema(
        name=message_cls.DESCRIPTOR.full_name,
        encoding="protobuf",
        data=build_file_descriptor_set(message_cls),
    )
    return writer.register_channel(topic=topic, message_encoding="protobuf", schema_id=schema_id)


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir).resolve()
    data_path = episode_dir / "data.npz"
    metadata_path = episode_dir / "metadata.json"
    output_path = Path(args.output).resolve() if args.output else episode_dir / f"{episode_dir.name}.mcap"

    metadata = load_metadata(metadata_path)
    data = np.load(data_path)

    timestamps = data["timestamp"]
    joint_position = data["joint_position"]
    ee_position = data["ee_position"]
    ee_orientation_wxyz = data["ee_orientation_wxyz"] if "ee_orientation_wxyz" in data else None
    ee_delta = data["ee_delta"]
    gripper_action = data["gripper_action"]
    gripper_open = data["gripper_open"]
    gripper_scalar = data["gripper_scalar"] if "gripper_scalar" in data else None
    object_position = data["object_position"] if "object_position" in data else None
    object_orientation = data["object_orientation_wxyz"] if "object_orientation_wxyz" in data else None

    top_dir = episode_dir / "images" / "top"
    wrist_dir = episode_dir / "images" / "wrist"
    top_files = sorted(top_dir.iterdir())
    wrist_files = sorted(wrist_dir.iterdir())

    if len(top_files) != len(timestamps) or len(wrist_files) != len(timestamps):
        raise RuntimeError("Image count does not match timestep count")

    joint_schema = {
        "type": "object",
        "properties": {
            "joint_names": {"type": "array", "items": {"type": "string"}},
            "joint_position": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["joint_names", "joint_position"],
    }
    gripper_schema = {
        "type": "object",
        "properties": {
            "gripper_action": {"type": "integer"},
            "gripper_open": {"type": "integer"},
            "gripper_scalar": {"type": ["number", "null"]},
        },
        "required": ["gripper_action", "gripper_open", "gripper_scalar"],
    }
    ee_delta_schema = {
        "type": "object",
        "properties": {
            "ee_delta": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["ee_delta"],
    }
    object_state_schema = {
        "type": "object",
        "properties": {
            "object_position": {"type": "array"},
            "object_orientation_wxyz": {"type": "array"},
        },
        "required": ["object_position", "object_orientation_wxyz"],
    }
    metadata_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "scene_name": {"type": "string"},
            "controller": {"type": "string"},
            "ee_frame": {"type": "string"},
            "num_steps": {"type": "integer"},
            "object_info": {"type": "array"},
        },
        "required": ["task", "scene_name", "controller", "ee_frame", "num_steps", "object_info"],
    }

    with output_path.open("wb") as f:
        writer = Writer(f)
        writer.start(profile="foxglove", library="isaac_table raw_to_mcap")

        top_channel = register_protobuf_schema(writer, "/camera/top/image/compressed", CompressedImage)
        wrist_channel = register_protobuf_schema(writer, "/camera/wrist/image/compressed", CompressedImage)
        pose_channel = register_protobuf_schema(writer, "/robot/ee_pose", PoseInFrame)
        joint_channel = register_json_schema(writer, "/robot/joint_state", "isaac_table.JointState", joint_schema)
        gripper_channel = register_json_schema(writer, "/robot/gripper_state", "isaac_table.GripperState", gripper_schema)
        action_channel = register_json_schema(writer, "/robot/action/ee_delta", "isaac_table.EEDelta", ee_delta_schema)
        object_channel = register_json_schema(writer, "/debug/object_state", "isaac_table.ObjectState", object_state_schema)
        metadata_channel = register_json_schema(writer, "/metadata/episode", "isaac_table.EpisodeMetadata", metadata_schema)

        first_time_ns = int(float(timestamps[0]) * 1_000_000_000)
        metadata_payload = {
            "task": metadata.get("task"),
            "scene_name": metadata.get("scene_name"),
            "controller": metadata.get("controller"),
            "ee_frame": metadata.get("ee_frame"),
            "num_steps": int(metadata.get("num_steps", len(timestamps))),
            "object_info": metadata.get("object_info", []),
        }
        writer.add_message(
            channel_id=metadata_channel,
            log_time=first_time_ns,
            publish_time=first_time_ns,
            data=json.dumps(metadata_payload, separators=(",", ":")).encode("utf-8"),
        )

        joint_names = [f"joint{i}" for i in range(1, int(joint_position.shape[1]) + 1)]

        for idx, ts in enumerate(timestamps):
            time_ns = int(float(ts) * 1_000_000_000)
            stamp = make_timestamp(float(ts))

            top_msg = CompressedImage()
            top_msg.timestamp.CopyFrom(stamp)
            top_msg.frame_id = "top_camera"
            top_msg.data = top_files[idx].read_bytes()
            top_msg.format = top_files[idx].suffix.lstrip(".")
            writer.add_message(top_channel, time_ns, top_msg.SerializeToString(), time_ns, sequence=idx)

            wrist_msg = CompressedImage()
            wrist_msg.timestamp.CopyFrom(stamp)
            wrist_msg.frame_id = "wrist_camera"
            wrist_msg.data = wrist_files[idx].read_bytes()
            wrist_msg.format = wrist_files[idx].suffix.lstrip(".")
            writer.add_message(wrist_channel, time_ns, wrist_msg.SerializeToString(), time_ns, sequence=idx)

            pose_msg = PoseInFrame()
            pose_msg.timestamp.CopyFrom(stamp)
            pose_msg.frame_id = "world"
            pose_msg.pose.position.x = float(ee_position[idx][0])
            pose_msg.pose.position.y = float(ee_position[idx][1])
            pose_msg.pose.position.z = float(ee_position[idx][2])
            if ee_orientation_wxyz is not None:
                pose_msg.pose.orientation.x = float(ee_orientation_wxyz[idx][1])
                pose_msg.pose.orientation.y = float(ee_orientation_wxyz[idx][2])
                pose_msg.pose.orientation.z = float(ee_orientation_wxyz[idx][3])
                pose_msg.pose.orientation.w = float(ee_orientation_wxyz[idx][0])
            writer.add_message(pose_channel, time_ns, pose_msg.SerializeToString(), time_ns, sequence=idx)

            joint_payload = {
                "joint_names": joint_names,
                "joint_position": joint_position[idx].tolist(),
            }
            writer.add_message(
                joint_channel,
                time_ns,
                json.dumps(joint_payload, separators=(",", ":")).encode("utf-8"),
                time_ns,
                sequence=idx,
            )

            gripper_payload = {
                "gripper_action": int(gripper_action[idx]),
                "gripper_open": int(gripper_open[idx]),
                "gripper_scalar": None if gripper_scalar is None else float(gripper_scalar[idx]),
            }
            writer.add_message(
                gripper_channel,
                time_ns,
                json.dumps(gripper_payload, separators=(",", ":")).encode("utf-8"),
                time_ns,
                sequence=idx,
            )

            action_payload = {
                "ee_delta": ee_delta[idx].tolist(),
            }
            writer.add_message(
                action_channel,
                time_ns,
                json.dumps(action_payload, separators=(",", ":")).encode("utf-8"),
                time_ns,
                sequence=idx,
            )

            if object_position is not None and object_orientation is not None:
                object_payload = {
                    "object_position": object_position[idx].tolist(),
                    "object_orientation_wxyz": object_orientation[idx].tolist(),
                }
                writer.add_message(
                    object_channel,
                    time_ns,
                    json.dumps(object_payload, separators=(",", ":")).encode("utf-8"),
                    time_ns,
                    sequence=idx,
                )

        writer.finish()

    print(f"[RawToMCAP] saved to {output_path}")


if __name__ == "__main__":
    main()
