"""
Sequential single-pass FFmpeg rawvideo exporter.
Pipes exact selected BGR24 frames into H.264 CRF 12 yuv420p MP4 files organized by target FPS.
"""

import os
import shutil
import subprocess
from typing import Dict, List, Optional
import cv2
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from src.config import PipelineConfig, SubClipSpec, VideoProperties


def ensure_ffmpeg_available() -> str:
    """Verifies that `ffmpeg` is installed and accessible on system PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg executable not found on system PATH. Please install FFmpeg before running this pipeline."
        )
    return ffmpeg_path


def create_output_directories(output_dir: str) -> Dict[int, str]:
    """
    Creates the output root directory and the three required framerate subdirectories:
    `08fps/`, `12fps/`, and `24fps/`.

    Returns:
        Dictionary mapping target_fps (8, 12, 24) to its absolute directory path.
    """
    fps_dirs: Dict[int, str] = {}
    for fps in (8, 12, 24):
        folder_path = os.path.join(output_dir, f"{fps:02d}fps")
        os.makedirs(folder_path, exist_ok=True)
        fps_dirs[fps] = folder_path
    return fps_dirs


def build_ffmpeg_command(
    width: int,
    height: int,
    target_fps: int,
    crf: int,
    output_filepath: str
) -> List[str]:
    """
    Builds the exact FFmpeg subprocess command required by STYLE_AND_RULES.md:
    libx264, -crf 12, -preset fast, -pix_fmt yuv420p, exact resolution and framerate, no audio.
    """
    return [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-framerate", str(target_fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-r", str(target_fps),
        output_filepath,
    ]


def export_subclips(
    video_path: str,
    subclips: List[SubClipSpec],
    video_props: VideoProperties,
    config: PipelineConfig,
    show_progress: bool = True
) -> List[str]:
    """
    Streams through the input video in a SINGLE sequential pass from frame 0 to the end,
    piping the selected raw BGR24 frames for each chronological SubClipSpec directly into FFmpeg.

    Args:
        video_path: Path to source video file.
        subclips: Chronological list of SubClipSpec objects.
        video_props: VideoProperties (width, height, fps, total_frames).
        config: PipelineConfig instance.
        show_progress: Whether to display a Rich progress bar.

    Returns:
        List of generated MP4 file paths in chronological order.
    """
    ensure_ffmpeg_available()
    fps_dirs = create_output_directories(config.output_dir)

    if not subclips:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video for export: {video_path}")

    width = video_props.width
    height = video_props.height
    exported_paths: List[str] = []

    current_frame_idx = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("Exporting H.264 CRF 12 Sub-Clips...", total=len(subclips))

        for subclip in subclips:
            out_dir = fps_dirs[subclip.target_fps]
            out_path = os.path.join(out_dir, subclip.filename)

            cmd = build_ffmpeg_command(
                width=width,
                height=height,
                target_fps=subclip.target_fps,
                crf=config.crf,
                output_filepath=out_path,
            )

            process: Optional[subprocess.Popen] = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            selected_set = set(subclip.selected_source_frames)
            max_needed_frame = (
                max(subclip.end_frame - 1, max(selected_set))
                if selected_set
                else subclip.end_frame - 1
            )

            try:
                while current_frame_idx <= max_needed_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if current_frame_idx in selected_set:
                        assert process.stdin is not None
                        process.stdin.write(frame.tobytes())
                    current_frame_idx += 1
            finally:
                if process.stdin:
                    process.stdin.close()

            stderr_output = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg failed exporting {subclip.filename} (exit code {return_code}):\n{stderr_output}"
                )

            exported_paths.append(out_path)
            progress.update(task, advance=1)

    cap.release()
    return exported_paths