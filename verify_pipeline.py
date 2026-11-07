"""
Automated end-to-end verification script (`verify_pipeline.py`).
Generates a synthetic 24fps .mkv anime test video containing:
  - Shot 1 (2.0s / 48 frames): Animated on Ones (changes every frame -> 24fps/Clip_0001_24fps.mp4)
  - Shot 2 (2.0s / 48 frames): Animated on Twos (changes every 2 frames -> 12fps/Clip_0002_12fps.mp4 with 24 unique frames @ 12fps)
  - Shot 3 (3.0s / 72 frames): 1.5s Static Hold (36 identical frames) followed immediately by
    1.5s Animated on Threes (changes every 3 frames) -> splits into:
      * 24fps/Clip_0003_24fps.mp4 (36 frames @ 24fps, reason="static_hold")
      * 08fps/Clip_0004_08fps.mp4 (12 unique frames @ 8fps, reason="clean_8fps")
Runs the full pipeline and verifies all output clips, framerates, frame counts, and zero-duplicate assertions.
"""

import json
import os
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
from rich.console import Console

from src.config import PipelineConfig
from src.frame_analyzer import preprocess_frame_for_diff, compute_frame_diff_score
from main import run_pipeline

console = Console()


def _draw_shot1_frame(drawing_idx: int, width: int, height: int) -> np.ndarray:
    """
    Shot 1 (Animated on Ones): Dark blue textured background with a bright yellow circle
    and sweeping vertical bar that moves significantly on every single frame.
    """
    img = np.full((height, width, 3), (60, 30, 15), dtype=np.uint8)
    # Add static background grid so PySceneDetect has rich luma/HSV content
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (90, 50, 30), 2)

    cx = int(60 + (drawing_idx * 11) % (width - 120))
    cy = int(80 + (drawing_idx * 7) % (height - 160))
    cv2.circle(img, (cx, cy), 45, (0, 230, 255), -1)
    bar_x = int((drawing_idx * 15) % (width - 30))
    cv2.rectangle(img, (bar_x, 20), (bar_x + 25, height - 20), (255, 200, 50), -1)
    return img


def _draw_shot2_frame(drawing_idx: int, width: int, height: int) -> np.ndarray:
    """
    Shot 2 (Animated on Twos): Bright crimson/magenta background (triggers hard cut from Shot 1)
    with a large green polygon/rectangle shifting position every new drawing.
    """
    img = np.full((height, width, 3), (30, 20, 180), dtype=np.uint8)
    for x in range(0, width, 45):
        cv2.line(img, (x, 0), (x, height), (60, 50, 220), 2)

    bx = int(40 + (drawing_idx * 22) % (width - 140))
    by = int(50 + (drawing_idx * 14) % (height - 140))
    cv2.rectangle(img, (bx, by), (bx + 90, by + 80), (50, 250, 80), -1)
    cv2.circle(img, (width - bx, height - by), 35, (255, 255, 255), -1)
    return img


def _draw_shot3_frame(drawing_idx: int, width: int, height: int) -> np.ndarray:
    """
    Shot 3 (1.5s Static Hold + 1.5s Animated on Threes): Bright teal/white background
    (triggers hard cut from Shot 2). When drawing_idx == 0 (held for 36 frames), scene is static.
    When drawing_idx >= 1, a bold orange/black character block moves across the screen on threes.
    """
    img = np.full((height, width, 3), (210, 220, 40), dtype=np.uint8)
    cv2.rectangle(img, (20, height - 80), (width - 20, height - 20), (140, 150, 20), -1)

    ox = int(70 + (drawing_idx * 35) % (width - 160))
    oy = int(70 + (drawing_idx * 19) % (height - 180))
    cv2.circle(img, (ox, oy), 50, (10, 30, 240), -1)
    cv2.rectangle(img, (ox - 30, oy - 30), (ox + 30, oy + 30), (20, 20, 20), -1)
    return img


def generate_synthetic_mkv(filepath: str, width: int = 640, height: int = 360, fps: int = 24) -> None:
    """
    Generates the synthetic 24fps .mkv test anime video using FFmpeg:
      - Shot 1 (48 frames / 2.0s): On Ones (48 drawings, hold=1)
      - Shot 2 (48 frames / 2.0s): On Twos (24 drawings, hold=2)
      - Shot 3 (72 frames / 3.0s):
          * First 36 frames (1.5s): Static Hold (1 drawing, hold=36)
          * Next 36 frames (1.5s): On Threes (12 drawings, hold=3)
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-crf", "10",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        filepath,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    # Shot 1: 48 frames on Ones (hold = 1)
    for i in range(48):
        frame = _draw_shot1_frame(i, width, height)
        proc.stdin.write(frame.tobytes())

    # Shot 2: 48 frames on Twos (24 unique drawings, each written twice)
    for drawing_idx in range(24):
        frame = _draw_shot2_frame(drawing_idx, width, height)
        for _ in range(2):
            proc.stdin.write(frame.tobytes())

    # Shot 3 Part A: 36 frames Static Hold (drawing_idx = 0 repeated 36 times)
    static_frame = _draw_shot3_frame(0, width, height)
    for _ in range(36):
        proc.stdin.write(static_frame.tobytes())

    # Shot 3 Part B: 36 frames on Threes (12 unique drawings, drawing_idx = 1..12, each written 3 times)
    for drawing_idx in range(1, 13):
        frame = _draw_shot3_frame(drawing_idx, width, height)
        for _ in range(3):
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    stderr_txt = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Failed to generate synthetic test MKV:\n{stderr_txt}")


def verify_exported_clip(
    clip_path: str,
    expected_fps: int,
    expected_frames: int,
    assert_zero_duplicates: bool,
    diff_threshold: float = 1.2
) -> None:
    """
    Opens an exported MP4 clip via OpenCV and verifies:
      1. File exists and is readable.
      2. Container framerate matches expected_fps.
      3. Decoded frame count matches expected_frames.
      4. If assert_zero_duplicates is True, every adjacent frame pair has diff_score >= diff_threshold.
    """
    assert os.path.isfile(clip_path), f"Expected exported clip does not exist: {clip_path}"

    cap = cv2.VideoCapture(clip_path)
    assert cap.isOpened(), f"OpenCV could not open exported clip: {clip_path}"

    actual_fps = round(float(cap.get(cv2.CAP_PROP_FPS)))
    assert actual_fps == expected_fps, (
        f"FPS mismatch for {clip_path}: expected {expected_fps}, got {actual_fps}"
    )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    assert len(frames) == expected_frames, (
        f"Frame count mismatch for {clip_path}: expected {expected_frames}, decoded {len(frames)}"
    )

    if assert_zero_duplicates:
        assert len(frames) >= 2, f"Expected at least 2 frames for RIFE interpolation in {clip_path}"
        prev_blur = preprocess_frame_for_diff(frames[0], target_width=640)
        for idx in range(1, len(frames)):
            curr_blur = preprocess_frame_for_diff(frames[idx], target_width=640)
            score = compute_frame_diff_score(prev_blur, curr_blur)
            assert score >= diff_threshold, (
                f"Duplicate frame found in {clip_path} between frames {idx - 1} and {idx} "
                f"(diff_score={score:.3f} < {diff_threshold})"
            )
            prev_blur = curr_blur


def main() -> None:
    """Runs the end-to-end verification test suite."""
    console.print("[bold cyan]=== Running End-to-End Verification Pipeline ===[/bold cyan]")
    work_dir = tempfile.mkdtemp(prefix="anime_cadence_verify_")

    try:
        synthetic_mkv = os.path.join(work_dir, "synthetic_test_anime.mkv")
        output_dir = os.path.join(work_dir, "output")

        console.print("1. Generating synthetic 24fps .mkv test anime video (168 frames / 7.0s)...")
        generate_synthetic_mkv(synthetic_mkv)

        config = PipelineConfig(
            input_path=synthetic_mkv,
            output_dir=output_dir,
            diff_threshold=1.2,
            scene_threshold=27.0,
            min_subclip_sec=0.5,
            static_hold_sec=1.0,
            crf=12,
        )

        console.print("2. Running Anime Cadence Splitter & Deduplicator pipeline...")
        subclips = run_pipeline(config=config, show_progress=True)

        console.print("3. Verifying exported clips, folder routing, framerates, and zero-duplicate guarantees...")
        assert len(subclips) == 4, f"Expected exactly 4 sub-clips, got {len(subclips)}"

        # Expected Clip 1: Shot 1 (Ones -> 24fps/Clip_0001_24fps.mp4, 48 frames @ 24fps)
        clip1_path = os.path.join(output_dir, "24fps", "Clip_0001_24fps.mp4")
        verify_exported_clip(
            clip_path=clip1_path,
            expected_fps=24,
            expected_frames=48,
            assert_zero_duplicates=True,
            diff_threshold=config.diff_threshold,
        )

        # Expected Clip 2: Shot 2 (Twos -> 12fps/Clip_0002_12fps.mp4, 24 unique frames @ 12fps)
        clip2_path = os.path.join(output_dir, "12fps", "Clip_0002_12fps.mp4")
        verify_exported_clip(
            clip_path=clip2_path,
            expected_fps=12,
            expected_frames=24,
            assert_zero_duplicates=True,
            diff_threshold=config.diff_threshold,
        )

        # Expected Clip 3: Shot 3 Part A (1.5s Static Hold -> 24fps/Clip_0003_24fps.mp4, 36 frames @ 24fps)
        clip3_path = os.path.join(output_dir, "24fps", "Clip_0003_24fps.mp4")
        verify_exported_clip(
            clip_path=clip3_path,
            expected_fps=24,
            expected_frames=36,
            assert_zero_duplicates=False,  # Static hold intentionally preserves identical frames at 24fps
            diff_threshold=config.diff_threshold,
        )
        assert subclips[2].reason == "static_hold", (
            f"Expected Clip 3 reason to be 'static_hold', got '{subclips[2].reason}'"
        )

        # Expected Clip 4: Shot 3 Part B (1.5s Threes -> 08fps/Clip_0004_08fps.mp4, 12 unique frames @ 8fps)
        clip4_path = os.path.join(output_dir, "08fps", "Clip_0004_08fps.mp4")
        verify_exported_clip(
            clip_path=clip4_path,
            expected_fps=8,
            expected_frames=12,
            assert_zero_duplicates=True,
            diff_threshold=config.diff_threshold,
        )
        assert subclips[3].reason == "clean_8fps", (
            f"Expected Clip 4 reason to be 'clean_8fps', got '{subclips[3].reason}'"
        )

        # Verify Manifest files
        json_manifest = os.path.join(output_dir, "timeline_manifest.json")
        csv_manifest = os.path.join(output_dir, "timeline_manifest.csv")
        assert os.path.isfile(json_manifest), "timeline_manifest.json was not created"
        assert os.path.isfile(csv_manifest), "timeline_manifest.csv was not created"

        with open(json_manifest, "r", encoding="utf-8") as jf:
            manifest_data = json.load(jf)
        assert manifest_data["summary"]["total_clips"] == 4
        assert manifest_data["summary"]["clips_24fps"] == 2
        assert manifest_data["summary"]["clips_12fps"] == 1
        assert manifest_data["summary"]["clips_08fps"] == 1

        console.print("\n[bold green]ALL VERIFICATION ASSERTIONS PASSED SUCCESSFULLY![/bold green]")
        console.print("  [green]✔[/green] 24fps/Clip_0001_24fps.mp4: 48 frames @ 24fps (Animated on Ones)")
        console.print("  [green]✔[/green] 12fps/Clip_0002_12fps.mp4: 24 unique frames @ 12fps (Zero duplicates verified)")
        console.print("  [green]✔[/green] 24fps/Clip_0003_24fps.mp4: 36 frames @ 24fps (1.5s Static Hold isolated)")
        console.print("  [green]✔[/green] 08fps/Clip_0004_08fps.mp4: 12 unique frames @ 8fps (Zero duplicates verified)")
        console.print("  [green]✔[/green] timeline_manifest.json & timeline_manifest.csv verified")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()