import numpy as np
import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import json
from icecream import ic
from dataclasses import dataclass, asdict
import cv2
from tqdm import tqdm


@dataclass
class FrameNS:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    file_path: str
    depth_file_path: str
    transform_matrix: list[list[float]]


@dataclass
class FrameSplatam:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    file_path: str
    depth_path: str
    transform_matrix: list[list[float]]


@dataclass
class NerfstudioTransform:
    camera_model: str
    orientation_override: str
    frames: list[FrameNS]


@dataclass
class SplatamTransform:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    frames: list[FrameSplatam]
    integer_depth_scale: float


def load_json_file(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def view_splatam_data(input_path: Path):
    transforms_json_path = input_path / "transforms.json"
    assert transforms_json_path.exists()
    transforms_data = load_json_file(transforms_json_path)
    # Convert the 'frames' list of dictionaries into a list of Frame instances
    frames = [FrameSplatam(**frame) for frame in transforms_data["frames"]]

    # Replace the 'frames' list of dictionaries in the data with the list of Frame instances
    transforms_data["frames"] = frames
    splatam_transforms = SplatamTransform(**transforms_data)

    integer_depth_scale = splatam_transforms.integer_depth_scale

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)
    for idx, frame in tqdm(enumerate(splatam_transforms.frames)):
        rr.set_time_sequence("world", idx)
        bgr = cv2.imread(str(input_path / frame.file_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(input_path / frame.depth_path), cv2.IMREAD_ANYDEPTH)
        # convert depth to meters, the png_depth_scale is 65535 / 10 = 6553.5
        # integer_depth_scale is the inverse or 1 / 6553.5 = 0.000152587890625
        depth = depth.astype(np.float32)  # * integer_depth_scale
        rr.log("world/camera/pinhole/image", rr.Image(rgb))
        rr.log(
            "world/camera/pinhole/depth",
            rr.DepthImage(depth, meter=6553.5),
        )
        rr.log(
            "world/camera/pinhole",
            rr.Pinhole(
                focal_length=(frame.fl_x, frame.fl_y),
                principal_point=(frame.cx, frame.cy),
                width=frame.w,
                height=frame.h,
                camera_xyz=rr.ViewCoordinates.RUB,
            ),
        )
        rotation = np.array(frame.transform_matrix)[:3, :3]
        translation = np.array(frame.transform_matrix)[:3, 3]
        rr.log(
            "world/camera/",
            rr.Transform3D(
                rr.TranslationAndMat3x3(translation=translation, mat3x3=rotation)
            ),
        )


def view_ns_data(input_path: Path):
    transforms_json_path = input_path / "transforms.json"
    assert transforms_json_path.exists()
    transforms_data = load_json_file(transforms_json_path)
    # Convert the 'frames' list of dictionaries into a list of Frame instances
    frames = [FrameNS(**frame) for frame in transforms_data["frames"]]

    # Replace the 'frames' list of dictionaries in the data with the list of Frame instances
    transforms_data["frames"] = frames
    ns_transforms = NerfstudioTransform(**transforms_data)

    rr.log("world", rr.ViewCoordinates.BRU, timeless=True)
    for idx, frame in tqdm(enumerate(ns_transforms.frames)):
        rr.set_time_sequence("world", idx)
        bgr = cv2.imread(str(input_path / frame.file_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(input_path / frame.depth_file_path), cv2.IMREAD_ANYDEPTH)
        ic(np.max(depth), np.min(depth))
        rr.log("world/camera/pinhole/image", rr.Image(rgb))
        rr.log("world/camera/pinhole/depth", rr.DepthImage(depth, meter=1000))
        rr.log(
            "world/camera/pinhole",
            rr.Pinhole(
                focal_length=(frame.fl_x, frame.fl_y),
                principal_point=(frame.cx, frame.cy),
                width=frame.w,
                height=frame.h,
                camera_xyz=rr.ViewCoordinates.RUB,
            ),
        )
        rotation = np.array(frame.transform_matrix)[:3, :3]
        translation = np.array(frame.transform_matrix)[:3, 3]
        rr.log(
            "world/camera/",
            rr.Transform3D(
                rr.TranslationAndMat3x3(translation=translation, mat3x3=rotation)
            ),
        )
        # if idx == 50:
        #     break


def convert_ns_to_splatam(input_path: Path):
    # replace -poly with -splatam
    name = input_path.name.replace("-poly", "-splatam")
    output_path = input_path.parent / name
    # make output_path
    output_path.mkdir(exist_ok=True)

    transforms_json_path = input_path / "transforms.json"
    assert transforms_json_path.exists()
    transforms_data = load_json_file(transforms_json_path)
    # Convert the 'frames' list of dictionaries into a list of Frame instances
    frames = [FrameNS(**frame) for frame in transforms_data["frames"]]

    # Replace the 'frames' list of dictionaries in the data with the list of Frame instances
    transforms_data["frames"] = frames
    ns_transforms = NerfstudioTransform(**transforms_data)

    # remove ./ from file_path and depth_file_path
    splatam_frames_list = []
    for frame in frames:
        file_path = frame.file_path[2:]
        depth_file_path = frame.depth_file_path[2:]
        # replace images with rgb for folder name
        file_path = file_path.replace("images", "rgb")
        # convert from .jpg to .png
        file_path = file_path.replace(".jpg", ".png")
        splatam_frames_list.append(
            FrameSplatam(
                fl_x=frame.fl_x,
                fl_y=frame.fl_y,
                cx=frame.cx,
                cy=frame.cy,
                w=frame.w,
                h=frame.h,
                file_path=file_path,
                depth_path=depth_file_path,
                transform_matrix=frame.transform_matrix,
            )
        )

    # convert ns_transforms to splatam_transforms
    splatam_transforms = SplatamTransform(
        fl_x=ns_transforms.frames[0].fl_x,
        fl_y=ns_transforms.frames[0].fl_y,
        cx=ns_transforms.frames[0].cx,
        cy=ns_transforms.frames[0].cy,
        w=ns_transforms.frames[0].w,
        h=ns_transforms.frames[0].h,
        frames=splatam_frames_list,
        # 16bit depth image is 0-65535, so divide by 10 to get meters
        integer_depth_scale=10 / 65535,
    )

    # write splatam_transforms to file
    splatam_transforms_json_path = output_path / "transforms.json"
    with open(splatam_transforms_json_path, "w") as file:
        json.dump(asdict(splatam_transforms), file)

    # copy images and depth images to output_path
    for idx, (splatam_frame, ns_frame) in tqdm(
        enumerate(zip(splatam_transforms.frames, ns_transforms.frames))
    ):
        rr.set_time_sequence("sequence", idx)
        depth_image = cv2.imread(
            str(input_path / ns_frame.depth_file_path), cv2.IMREAD_ANYDEPTH
        )
        rr.log("sequence/depth", rr.DepthImage(depth_image, meter=1000))
        # convert depth from 1000 (mm) -> 1m
        depth_image = depth_image.astype(np.float32) * 0.001
        # convert depth to uint16 0-65535 (divide by 10 to match splatam depth scale)
        depth_image = (depth_image * 65535 / 10).astype(np.uint16)
        rr.log("sequence/depth_new", rr.DepthImage(depth_image, meter=6553.5))
        # save depth image
        depth_image_path = output_path / splatam_frame.depth_path
        depth_image_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(depth_image_path), depth_image)
        # copy rgb image to new path
        rgb_image_path = output_path / splatam_frame.file_path
        rgb_image_path.parent.mkdir(exist_ok=True, parents=True)

        image = cv2.imread(str(input_path / ns_frame.file_path))
        cv2.imwrite(str(rgb_image_path), image)


def main(input_path: Path, data_type: str, convert: bool):
    if convert:
        convert_ns_to_splatam(input_path)
        return

    if data_type == "splatam":
        view_splatam_data(input_path)
    elif data_type == "ns":
        view_ns_data(input_path)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--data-type", default="ns", choices=["splatam", "ns"])
    parser.add_argument("--convert", action="store_true", help="Convert ns to splatam")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "view splat")
    main(args.input_path, args.data_type, args.convert)
    rr.script_teardown(args)
