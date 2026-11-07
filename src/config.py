"""
Configuration and core data structures for the Anime Cadence Splitter & Deduplicator.
"""

from dataclasses import dataclass, field
from typing import List, Literal

CadenceFPS = Literal[8, 12, 24]


@dataclass
class PipelineConfig:
    """Global configuration parameters for the video analysis and export pipeline."""
    input_path: str
    output_dir: str
    diff_threshold: float = 1.2          # MAD/p98 threshold (0-255 scale) below which frames are duplicates
    scene_threshold: float = 27.0        # PySceneDetect ContentDetector threshold
    min_subclip_sec: float = 0.5         # Minimum duration (in seconds) for a sub-clip split (12 frames @ 24fps)
    static_hold_sec: float = 1.0         # Duration (in seconds) of no motion to force 24fps static hold (24 frames)
    crf: int = 12                        # H.264 CRF compression quality (12 = near lossless)
    analysis_width: int = 640            # Downscaled width used only for fast MAD diff calculation
    strict_duration_match: bool = True   # Subsample mixed 2s/3s fallback to match exact real-time duration at 8fps


@dataclass
class VideoProperties:
    """Metadata properties of the source video file."""
    width: int
    height: int
    fps: float
    total_frames: int


@dataclass
class ShotBounds:
    """Frame boundaries for a single continuous camera shot."""
    shot_index: int
    start_frame: int                     # Inclusive (0-indexed)
    end_frame: int                       # Exclusive (0-indexed)

    @property
    def num_frames(self) -> int:
        """Total number of source frames in this shot."""
        return self.end_frame - self.start_frame


@dataclass
class ShotAnalysis:
    """Per-frame difference scores and new-drawing flags for a single shot."""
    shot: ShotBounds
    diff_scores: List[float]             # Length = (end_frame - start_frame), index 0 is always 999.0 (new frame)
    is_new_drawing: List[bool]           # True if diff_score >= diff_threshold (always True at index 0)


@dataclass
class DrawingRun:
    """Run-Length Encoded representation of a single unique drawing held over 1 or more frames."""
    frame_offset: int                    # Index relative to shot start (0 <= k < L)
    abs_frame: int                       # Exact frame index in the source video (start_frame + k)
    hold_length: int                     # Number of consecutive source frames this drawing stays on screen


@dataclass
class SubClipSpec:
    """Specification for an exported sub-clip assigned to a specific framerate bucket."""
    clip_id: int                         # 1-based global chronological index (0001, 0002...)
    parent_shot_index: int
    start_frame: int                     # Inclusive source frame index
    end_frame: int                       # Exclusive source frame index
    target_fps: CadenceFPS               # 8, 12, or 24
    reason: str                          # e.g., "static_hold", "clean_24fps", "clean_12fps", "clean_8fps", "mixed_fallback_8fps"
    selected_source_frames: List[int] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Standardized chronological filename."""
        return f"Clip_{self.clip_id:04d}_{self.target_fps:02d}fps.mp4"

    @property
    def folder_name(self) -> str:
        """Target FPS subfolder name."""
        return f"{self.target_fps:02d}fps"
