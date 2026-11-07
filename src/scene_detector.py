"""
Scene cut detection wrapper using PySceneDetect ContentDetector and OpenCV.
"""

import os
from typing import List, Tuple
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from rich.console import Console

from src.config import ShotBounds, VideoProperties

console = Console()


def inspect_video(video_path: str) -> VideoProperties:
    """
    Inspects input video properties via cv2.VideoCapture.

    Args:
        video_path: Path to the source video file.

    Returns:
        VideoProperties containing width, height, fps, and actual countable frames.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions ({width}x{height}) in {video_path}")
    if fps <= 0:
        fps = 24.0

    return VideoProperties(
        width=width,
        height=height,
        fps=fps,
        total_frames=max(0, reported_frames),
    )


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
    show_progress: bool = True
) -> Tuple[List[ShotBounds], VideoProperties]:
    """
    Detects hard cuts in the video using PySceneDetect's ContentDetector.
    Falls back gracefully to a single shot [0, total_frames) if no cuts are found.

    Args:
        video_path: Path to the input .mkv or .mp4 file.
        threshold: ContentDetector threshold (default 27.0).
        show_progress: Whether to display progress during scene detection.

    Returns:
        Tuple of (List[ShotBounds], VideoProperties).
    """
    props = inspect_video(video_path)

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video=video, show_progress=show_progress)
    scene_list = scene_manager.get_scene_list()

    shots: List[ShotBounds] = []
    if not scene_list:
        if props.total_frames > 0:
            shots.append(ShotBounds(shot_index=0, start_frame=0, end_frame=props.total_frames))
    else:
        for idx, (start_tc, end_tc) in enumerate(scene_list):
            s_frame = int(start_tc.get_frames())
            e_frame = int(end_tc.get_frames())
            if e_frame > s_frame:
                shots.append(ShotBounds(shot_index=idx, start_frame=s_frame, end_frame=e_frame))

        # Update total_frames from the final scene bound if PySceneDetect decoded more accurately than container header
        if shots:
            props.total_frames = max(props.total_frames, shots[-1].end_frame)

    # Ensure contiguous coverage starting from 0
    if not shots and props.total_frames > 0:
        shots = [ShotBounds(shot_index=0, start_frame=0, end_frame=props.total_frames)]

    return shots, props
