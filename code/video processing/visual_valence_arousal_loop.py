"""
Runs EmoNet (Toisoul et al. 2021) on campaign videos to extract per-frame
continuous valence/arousal and discrete emotion, producing the *_va.csv
files that feature_extract_valenceArousal.py aggregates to per-video
statistics.

Source / attribution: adapted from demo_video.py in the official EmoNet
repository, https://github.com/face-analysis/emonet, by Jean Kossaifi,
Antoine Toisoul, and Adrian Bulat. That repository (code and pretrained
weights) is distributed under a Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International licence
(CC BY-NC-ND 4.0) -- see https://github.com/face-analysis/emonet for the
license file and citation details. This script is an adapted (modified)
version of their demo_video.py for batch-processing this project's videos;
non-commercial use only.

Requires:
- Pretrained EmoNet weights, NOT included in this repository. Download
  emonet_5.pth / emonet_8.pth from the official repo
  (https://github.com/face-analysis/emonet, see its README for the
  download link) and place them in a `pretrained/` folder next to this
  script (i.e. code/video processing/pretrained/emonet_8.pth).
- The `emonet` package shipped alongside this script (code/video
  processing/emonet/) must be importable -- run this script from within
  `code/video processing/`, or add that directory to PYTHONPATH.
- pip install face-alignment (for the SFD face detector).
"""
from typing import List, Dict
from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm, trange
import torch
from torch import nn
from skimage import io
from face_alignment.detection.sfd.sfd_detector import SFDDetector

from emonet.models import EmoNet

import cv2


def load_video_1sec(video_path: Path) -> List[np.ndarray]:
    """
    Loads a video using OpenCV.
    """
    video_capture = cv2.VideoCapture(str(video_path))
    list_frames_rgb = []

    if not video_capture.isOpened():
        print("Error: Cannot open video file.")
        return list_frames_rgb

    # Get video FPS
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps)  # Number of frames to skip to achieve 1 frame per second

    frame_idx = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        # Only process every 'frame_interval' frames
        if frame_idx % frame_interval == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_frames_rgb.append(image_rgb)

        frame_idx += 1

    video_capture.release()
    return list_frames_rgb

def load_video(video_path: Path) -> List[np.ndarray]:
    """
    Loads a video using OpenCV.
    """
    video_capture = cv2.VideoCapture(video_path)

    list_frames_rgb = []

    # Reads all the frames
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        list_frames_rgb.append(image_rgb)

    return list_frames_rgb

def load_emonet(n_expression: int, device: str):
    """
    Loads the emotion recognition model.
    """

    # Loading the model
    state_dict_path = Path(__file__).parent.joinpath(
        "pretrained", f"emonet_{n_expression}.pth"
    )

    print(f"Loading the emonet model from {state_dict_path}.")
    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    return net


def run_emonet(
    emonet: torch.nn.Module, frame_rgb: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Runs the emotion recognition model on a single frame.
    """
    # Resize image to (256,256)
    image_rgb = cv2.resize(frame_rgb, (image_size, image_size))

    # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
    image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(device) / 255.0

    with torch.no_grad():
        output = emonet(image_tensor.unsqueeze(0))

    return output


def plot_valence_arousal(
    valence: float, arousal: float, circumplex_size=512
) -> np.ndarray:
    """
    Assumes valence and arousal in range [-1;1].
    """
    circumplex_path = Path(__file__).parent / "images/circumplex.png"

    circumplex_image = cv2.imread(circumplex_path)
    circumplex_image = cv2.resize(circumplex_image, (circumplex_size, circumplex_size))

    # Position in range [0,circumplex_size/2] - arousal axis goes up, so need to take the opposite
    position = (
        (valence + 1.0) / 2.0 * circumplex_size,
        (1.0 - arousal) / 2.0 * circumplex_size,
    )

    cv2.circle(
        circumplex_image, (int(position[0]), int(position[1])), 16, (0, 0, 255), -1
    )

    return circumplex_image


def make_visualization(
    frame_rgb: np.ndarray,
    face_crop_rgb: np.ndarray,
    face_bbox: torch.Tensor,
    emotion_prediction: Dict[str, torch.Tensor],
    font_scale=2,
) -> np.ndarray:
    """
    Composes the final visualization with detected face, landmarks, discrete and continuous emotions.
    """
    # Visualize the detected face
    cv2.rectangle(
        frame_rgb,
        (face_bbox[0], face_bbox[1]),
        (face_bbox[2], face_bbox[3]),
        (255, 0, 0),
        8,
    )

    # Add the discrete emotion next to it
    predicted_emotion_class_idx = (
        torch.argmax(nn.functional.softmax(emotion_prediction["expression"], dim=1))
        .cpu()
        .item()
    )
    frame_rgb = cv2.putText(
        frame_rgb,
        emotion_classes[predicted_emotion_class_idx],
        ((face_bbox[0] + face_bbox[2]) // 2, face_bbox[1] + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    print("emotion_classes",emotion_classes[predicted_emotion_class_idx])

    # Landmarks visualization
    # Resize to the original face_crop image size
    heatmap = torch.nn.functional.interpolate(
        emotion_prediction["heatmap"],
        (face_crop_rgb.shape[0], face_crop_rgb.shape[1]),
        mode="bilinear",
    )

    landmark_visualization = face_crop_rgb.copy()
    for landmark_idx in range(heatmap[0].shape[0]):
        # Detect the position of each landmark and draw a circle there
        landmark_position = (
            heatmap[0, landmark_idx, :, :] == torch.max(heatmap[0, landmark_idx, :, :])
        ).nonzero()
        cv2.circle(
            landmark_visualization,
            (
                int(landmark_position[0][1].cpu().item()),
                int(landmark_position[0][0].cpu().item()),
            ),
            4,
            (255, 255, 255),
            -1,
        )

    # Valence and arousal visualization
    circumplex_bgr = plot_valence_arousal(
        emotion_prediction["valence"].clamp(-1.0, 1.0),
        emotion_prediction["arousal"].clamp(-1.0, 1.0),
        frame_rgb.shape[0],
    )
    print("valence",emotion_prediction["valence"])
    print("arousal", emotion_prediction["arousal"])

    # Compose the final visualization
    visualization = np.zeros(
        (frame_rgb.shape[0], frame_rgb.shape[1] + frame_rgb.shape[0] // 2, 3),
        dtype=np.uint8,
    )

    # Resize the circumplex and face crop to match the frame size
    circumplex_bgr = cv2.resize(
        circumplex_bgr, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    landmark_visualization = cv2.resize(
        landmark_visualization, (frame_rgb.shape[0] // 2, frame_rgb.shape[0] // 2)
    )
    visualization[:, : frame_rgb.shape[1], :] = frame_rgb[:, :, ::-1].astype(np.uint8)
    visualization[
        : frame_rgb.shape[0] // 2, frame_rgb.shape[1] :, :
    ] = landmark_visualization[:, :, ::-1].astype(
        np.uint8
    )
    visualization[frame_rgb.shape[0] // 2 :, frame_rgb.shape[1] :, :] = (
        circumplex_bgr.astype(np.uint8)
    )

    return visualization, emotion_classes[predicted_emotion_class_idx], emotion_prediction["valence"].item(), emotion_prediction["arousal"].item()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nclasses",
        type=int,
        default=8,
        choices=[5, 8],
        help="Number of emotional classes to test the model on. Please use 5 or 8.",
    )
    parser.add_argument(
        "--oneSecond",
        action="store_true",
        help="If set, extracts one frame per second from the video. Default is False.",
    )

    parser.add_argument(
        "--input_root",
        type=str,
        default=os.environ.get("INPUT_VIDEO_ROOT", "./data/ig_videos_validRaw"),
        help="Path where the input videos are loaded from (a folder of .mp4 files).",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.environ.get("OUTPUT_CSV_DIR", "./output/ig_videos_va_csv"),
        help="Path where the output csv is saved.",
    )
    parser.add_argument(
        "--output_mp4",
        type=str,
        default=os.environ.get("OUTPUT_MP4_DIR", "./output/ig_videos_va_mp4"),
        help="Path where the output video is saved.",
    )
    parser.add_argument(
        "--exclude_files",
        type=str,
        default=os.environ.get("EXCLUDE_VIDEO_FILES", ""),
        help="Comma-separated list of video file paths to skip (e.g. known-corrupt files).",
    )
    args = parser.parse_args()

    # Parameters of the experiments
    n_expression = args.nclasses
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_size = 256
    emotion_classes = {
        0: "Neutral",
        1: "Happy",
        2: "Sad",
        3: "Surprise",
        4: "Fear",
        5: "Disgust",
        6: "Anger",
        7: "Contempt",
    }

    print(f"Loading emonet")
    emonet = load_emonet(n_expression, device)

    print(f"Loading face detector")
    sfd_detector = SFDDetector(device)

    print(f"Loading video")
    print(args.input_root+r"/*mp4")

    input_root = args.input_root
    folder_path = glob.glob(input_root + "/*.mp4")

    exclude_files = {f.strip() for f in args.exclude_files.split(",") if f.strip()}
    if exclude_files:
        folder_path = [file for file in folder_path if file not in exclude_files]

    for video_path  in tqdm(folder_path):
        print("video_path", video_path)

        if args.oneSecond:
            print("Extracting one frame per second.")
            list_frames_rgb = load_video_1sec(Path(video_path))

            os.makedirs(Path(args.output_mp4+"_oneSecond"), exist_ok=True)
            os.makedirs(Path(args.output_csv+"_oneSecond"), exist_ok=True)
            output_video_path = Path(args.output_mp4+"_oneSecond") / f"{Path(video_path).stem}_va.mp4"
            output_csv_path = Path(args.output_csv+"_oneSecond") / f"{Path(video_path).stem}_va.csv"

        else:
            print("Processing all frames.")
            list_frames_rgb = load_video(Path(video_path))

            os.makedirs(Path(args.output_mp4+"_fps24"), exist_ok=True)
            os.makedirs(Path(args.output_csv+"_fps24"), exist_ok=True)
            output_video_path = Path(args.output_mp4+"_fps24") / f"{Path(video_path).stem}_fps24_va.mp4"
            output_csv_path = Path(args.output_csv+"_fps24") / f"{Path(video_path).stem}_fps24_va.csv"

        if os.path.exists(output_video_path):
            print(f"Skipping {output_video_path}, output already exists.")
            continue

        visualization_frames = []

        va_df = pd.DataFrame(columns=["time", "emotion", "valence", "arousal"])

        for i, frame in enumerate(list_frames_rgb):

            # Run face detector
            with torch.no_grad():
                # Face detector requires BGR frame
                detected_faces = sfd_detector.detect_from_image(frame[:, :, ::-1])

            # If at least a face has been detected, run emotion recognition on the first face
            if len(detected_faces) > 0:
                bbox = np.array(detected_faces[0]).astype(np.int32)

                bbox[0] = max(bbox[0], 0)
                bbox[1] = max(bbox[1], 0)
                bbox[2] = min(bbox[2], frame.shape[1])  # W
                bbox[3] = min(bbox[3], frame.shape[0])  # H

                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

                    if face_crop.size > 0:
                        emotion_prediction = run_emonet(emonet, face_crop.copy())
                        visualization_bgr, emotion, valence, arousal = make_visualization(
                            frame.copy(), face_crop.copy(), bbox, emotion_prediction
                        )
                        visualization_frames.append(visualization_bgr)

                        if args.oneSecond:
                            minutes, seconds = divmod(i, 60)
                        else:
                            fps = 24
                            total_seconds = i / fps
                            minutes, seconds = divmod(int(total_seconds), 60)

                        time_convert = f"{minutes:02}:{seconds:02}"
                        new_row = pd.DataFrame({
                            "time": [time_convert],
                            "emotion": [emotion],
                            "valence": [valence],
                            "arousal": [arousal]
                        })
                        va_df = pd.concat([va_df, new_row], ignore_index=True)
                    else:
                        print(f"Warning: Empty face crop for frame {i}.")
                else:
                    print(f"Warning: Invalid bounding box for frame {i}.")
            else:
                print(f"No face detected in frame {i}.")
                visualization = np.zeros(
                    (frame.shape[0], frame.shape[1] + frame.shape[0] // 2, 3),
                    dtype=np.uint8,
                )
                visualization[:, :frame.shape[1], :] = frame[:, :, ::-1].astype(np.uint8)
                visualization_frames.append(visualization)

            print(f"Ran prediction on {i}/{len(list_frames_rgb)} frames")

        # Write the result as a video
        if visualization_frames:
            out = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                24.0,
                (visualization_frames[0].shape[1], visualization_frames[0].shape[0]),
            )

            for frame in visualization_frames:
                out.write(frame)

        va_df.to_csv(str(output_csv_path), index=False)
        print(f"Finished processing {video_path}")
