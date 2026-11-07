"""
Sequential single-pass frame difference analyzer.
Computes downscaled blurred grayscale MAD + 98th percentile diff scores across all shots.
"""

from typing import List, Optional
import cv2
import numpy as np
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from src.config import PipelineConfig, ShotBounds, ShotAnalysis


def preprocess_frame_for_diff(frame: np.ndarray, target_width: int = 640) -> np.ndarray:
    """
    Converts a BGR frame to grayscale, resizes to target_width with proportional height
    using cv2.INTER_AREA, and applies a 3x3 Gaussian blur to suppress macroblock noise.

    Args:
        frame: Raw BGR image from OpenCV.
        target_width: Downscaled width (default 640).

    Returns:
        Blurred grayscale numpy array.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w != target_width and w > 0:
        scale = target_width / float(w)
        target_height = max(1, int(round(h * scale)))
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        resized = gray

    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    return blurred


def compute_frame_diff_score(blur_prev: np.ndarray, blur_curr: np.ndarray) -> float:
    """
    Calculates the effective frame difference metric combining Mean Absolute Difference (MAD)
    and the 98th Percentile Absolute Difference:
        score = max(mad, p98 * 0.25)

    Args:
        blur_prev: Preprocessed previous frame.
        blur_curr: Preprocessed current frame.

    Returns:
        Floating-point difference score on a 0-255 scale.
    """
    abs_diff = cv2.absdiff(blur_prev, blur_curr)
    mad = float(np.mean(abs_diff))
    p98 = float(np.percentile(abs_diff, 98))
    return max(mad, p98 * 0.25)


def analyze_shots(
    video_path: str,
    shots: List[ShotBounds],
    config: PipelineConfig,
    show_progress: bool = True
) -> List[ShotAnalysis]:
    """
    Streams through the input video sequentially in a single pass, computing per-frame
    difference scores and `is_new_drawing` flags for every shot without storing full frames in RAM.

    Args:
        video_path: Path to the source video file.
        shots: Chronological list of ShotBounds from scene detection.
        config: PipelineConfig instance.
        show_progress: Whether to display a Rich progress bar.

    Returns:
        List of ShotAnalysis objects (one per non-empty shot).
    """
    if not shots:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for frame analysis: {video_path}")

    total_expected_frames = shots[-1].end_frame
    analyses: List[ShotAnalysis] = []

    shot_idx = 0
    num_shots = len(shots)
    current_shot = shots[0]
    curr_diff_scores: List[float] = []
    curr_is_new: List[bool] = []
    prev_blurred: Optional[np.ndarray] = None

    frame_idx = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("Analyzing Frame Differences...", total=total_expected_frames)

        while shot_idx < num_shots:
            ret, frame = cap.read()
            if not ret:
                break

            # Advance shot pointer if frame_idx has reached or passed current_shot.end_frame
            while shot_idx < num_shots and frame_idx >= current_shot.end_frame:
                if curr_diff_scores:
                    actual_bounds = ShotBounds(
                        shot_index=current_shot.shot_index,
                        start_frame=current_shot.start_frame,
                        end_frame=current_shot.start_frame + len(curr_diff_scores),
                    )
                    analyses.append(
                        ShotAnalysis(
                            shot=actual_bounds,
                            diff_scores=curr_diff_scores,
                            is_new_drawing=curr_is_new,
                        )
                    )
                shot_idx += 1
                if shot_idx < num_shots:
                    current_shot = shots[shot_idx]
                    curr_diff_scores = []
                    curr_is_new = []
                    prev_blurred = None

            if shot_idx >= num_shots:
                break

            if frame_idx >= current_shot.start_frame:
                curr_blurred = preprocess_frame_for_diff(frame, target_width=config.analysis_width)

                if frame_idx == current_shot.start_frame or prev_blurred is None:
                    score = 999.0
                    is_new = True
                else:
                    score = compute_frame_diff_score(prev_blurred, curr_blurred)
                    is_new = bool(score >= config.diff_threshold)

                curr_diff_scores.append(score)
                curr_is_new.append(is_new)
                prev_blurred = curr_blurred

            frame_idx += 1
            progress.update(task, advance=1)

    # Flush final shot if video ended before or at current_shot.end_frame
    if shot_idx < num_shots and curr_diff_scores:
        actual_bounds = ShotBounds(
            shot_index=current_shot.shot_index,
            start_frame=current_shot.start_frame,
            end_frame=current_shot.start_frame + len(curr_diff_scores),
        )
        analyses.append(
            ShotAnalysis(
                shot=actual_bounds,
                diff_scores=curr_diff_scores,
                is_new_drawing=curr_is_new,
            )
        )

    cap.release()
    return analyses