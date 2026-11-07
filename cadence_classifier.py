"""
Cadence classification, shot segmentation, RLE drawing hold analysis,
and unique frame selection with zero-duplicate verification.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.config import (
    CadenceFPS,
    DrawingRun,
    PipelineConfig,
    ShotAnalysis,
    SubClipSpec,
)


@dataclass
class _RunSegment:
    """Internal helper representing a contiguous group of DrawingRuns."""
    runs: List[DrawingRun]
    candidate_fps: CadenceFPS
    is_stable: bool = False
    reason: str = ""

    @property
    def start_frame(self) -> int:
        return self.runs[0].abs_frame

    @property
    def end_frame(self) -> int:
        last = self.runs[-1]
        return last.abs_frame + last.hold_length

    @property
    def total_frames(self) -> int:
        return sum(r.hold_length for r in self.runs)


def compute_drawing_runs(analysis: ShotAnalysis) -> List[DrawingRun]:
    """
    Performs Run-Length Encoding (RLE) on `analysis.is_new_drawing` for a shot.

    Args:
        analysis: ShotAnalysis containing `is_new_drawing` boolean flags.

    Returns:
        List of DrawingRun objects covering the entire shot.
    """
    flags = analysis.is_new_drawing
    start_frame = analysis.shot.start_frame
    length = len(flags)
    if length == 0:
        return []

    runs: List[DrawingRun] = []
    current_offset = 0

    for k in range(1, length):
        if flags[k]:
            runs.append(
                DrawingRun(
                    frame_offset=current_offset,
                    abs_frame=start_frame + current_offset,
                    hold_length=k - current_offset,
                )
            )
            current_offset = k

    # Append final run reaching the end of the shot
    runs.append(
        DrawingRun(
            frame_offset=current_offset,
            abs_frame=start_frame + current_offset,
            hold_length=length - current_offset,
        )
    )
    return runs


def raw_local_cadence(hold_length: int) -> CadenceFPS:
    """
    Maps a single DrawingRun's hold_length to its candidate local cadence:
      hold_length == 1 -> 24 fps
      hold_length == 2 -> 12 fps
      hold_length >= 3 -> 8 fps
    """
    if hold_length <= 1:
        return 24
    if hold_length == 2:
        return 12
    return 8


def classify_single_interval(runs: List[DrawingRun]) -> Tuple[CadenceFPS, str]:
    """
    Applies Step 4 (Single-Interval Classification Rule) on a sequence of DrawingRuns.

    Decision Tree:
      1. F1 / T >= 0.70 -> 24 fps ("clean_24fps")
      2. F2 / T >= 0.75 and F3 == 0 -> 12 fps ("clean_12fps")
      3. F2 / T >= 0.80 -> 12 fps ("dominant_12fps")
      4. F3 / T >= 0.70 -> 8 fps ("clean_8fps")
      5. Otherwise -> 8 fps ("mixed_fallback_8fps")
    """
    total_frames = sum(r.hold_length for r in runs)
    if total_frames <= 0:
        return 24, "clean_24fps"

    f1 = sum(r.hold_length for r in runs if r.hold_length == 1)
    f2 = sum(r.hold_length for r in runs if r.hold_length == 2)
    f3 = sum(r.hold_length for r in runs if r.hold_length >= 3)

    if f1 / total_frames >= 0.70:
        return 24, "clean_24fps"
    if (f2 / total_frames >= 0.75) and (f3 == 0):
        return 12, "clean_12fps"
    if f2 / total_frames >= 0.80:
        return 12, "dominant_12fps"
    if f3 / total_frames >= 0.70:
        return 8, "clean_8fps"
    return 8, "mixed_fallback_8fps"


def segment_moving_interval(
    runs: List[DrawingRun],
    min_subclip_frames: int
) -> List[Tuple[List[DrawingRun], CadenceFPS, str]]:
    """
    Implements Step 2 & Step 3 (Windowed Segmentation & Minimum Sub-Clip Length)
    for a non-static moving interval.

    Args:
        runs: Contiguous DrawingRuns in a moving interval.
        min_subclip_frames: Minimum source frames required for a split sub-clip (default 12).

    Returns:
        List of tuples: (segment_runs, target_fps, reason).
    """
    if not runs:
        return []

    total_interval_frames = sum(r.hold_length for r in runs)

    # If the entire moving interval is shorter than 2 * min_subclip_frames (< 24 frames),
    # it cannot be split into two valid >= 12-frame sub-clips.
    if total_interval_frames < 2 * min_subclip_frames:
        fps, reason = classify_single_interval(runs)
        return [(runs, fps, reason)]

    # 1. Group contiguous runs that share the same candidate cadence (24, 12, or 8) into raw segments
    raw_segments: List[_RunSegment] = []
    for run in runs:
        cand = raw_local_cadence(run.hold_length)
        if raw_segments and raw_segments[-1].candidate_fps == cand:
            raw_segments[-1].runs.append(run)
        else:
            raw_segments.append(_RunSegment(runs=[run], candidate_fps=cand))

    # 2. Identify stable segments (>= min_subclip_frames) vs unstable micro-segments (< min_subclip_frames)
    for seg in raw_segments:
        seg.is_stable = seg.total_frames >= min_subclip_frames
        if seg.is_stable:
            seg.candidate_fps, seg.reason = classify_single_interval(seg.runs)

    # If no stable segments exist at all or only one segment exists, classify consecutive merged unstable
    # or the whole interval directly
    merged_pass: List[_RunSegment] = []
    i = 0
    while i < len(raw_segments):
        if raw_segments[i].is_stable:
            merged_pass.append(raw_segments[i])
            i += 1
        else:
            # 4. Merge consecutive unstable micro-segments together
            unstable_runs: List[DrawingRun] = []
            while i < len(raw_segments) and not raw_segments[i].is_stable:
                unstable_runs.extend(raw_segments[i].runs)
                i += 1

            u_frames = sum(r.hold_length for r in unstable_runs)
            if u_frames >= min_subclip_frames:
                fps, reason = classify_single_interval(unstable_runs)
                merged_pass.append(
                    _RunSegment(
                        runs=unstable_runs,
                        candidate_fps=fps,
                        is_stable=True,
                        reason=reason,
                    )
                )
            else:
                merged_pass.append(
                    _RunSegment(
                        runs=unstable_runs,
                        candidate_fps=8,
                        is_stable=False,
                        reason="unstable",
                    )
                )

    # Absorb any remaining unstable micro-segments (< min_subclip_frames) into adjacent stable segments
    # (preferring the lower FPS neighbor if between 12 and 8, or matching the dominant neighbor)
    while any(not seg.is_stable for seg in merged_pass) and len(merged_pass) > 1:
        for idx, seg in enumerate(merged_pass):
            if not seg.is_stable:
                if idx == 0:
                    target_idx = 1
                elif idx == len(merged_pass) - 1:
                    target_idx = idx - 1
                else:
                    left_seg = merged_pass[idx - 1]
                    right_seg = merged_pass[idx + 1]
                    # Prefer lower FPS neighbor if between 12 and 8, otherwise dominant neighbor
                    if {left_seg.candidate_fps, right_seg.candidate_fps} == {8, 12}:
                        target_idx = (idx - 1) if left_seg.candidate_fps == 8 else (idx + 1)
                    elif left_seg.total_frames >= right_seg.total_frames:
                        target_idx = idx - 1
                    else:
                        target_idx = idx + 1

                if target_idx < idx:
                    merged_pass[target_idx].runs.extend(seg.runs)
                    merged_pass[target_idx].candidate_fps, merged_pass[target_idx].reason = (
                        classify_single_interval(merged_pass[target_idx].runs)
                    )
                    merged_pass[target_idx].is_stable = True
                    merged_pass.pop(idx)
                else:
                    merged_pass[target_idx].runs = seg.runs + merged_pass[target_idx].runs
                    merged_pass[target_idx].candidate_fps, merged_pass[target_idx].reason = (
                        classify_single_interval(merged_pass[target_idx].runs)
                    )
                    merged_pass[target_idx].is_stable = True
                    merged_pass.pop(idx)
                break

    # Merge adjacent segments that ended up with the same target_fps
    consolidated: List[_RunSegment] = []
    for seg in merged_pass:
        fps, reason = classify_single_interval(seg.runs)
        seg.candidate_fps = fps
        seg.reason = reason
        if consolidated and consolidated[-1].candidate_fps == seg.candidate_fps:
            consolidated[-1].runs.extend(seg.runs)
            consolidated[-1].candidate_fps, consolidated[-1].reason = classify_single_interval(
                consolidated[-1].runs
            )
        else:
            consolidated.append(seg)

    return [(seg.runs, seg.candidate_fps, seg.reason) for seg in consolidated]


def select_frames_for_subclip(
    runs: List[DrawingRun],
    start_frame: int,
    end_frame: int,
    target_fps: CadenceFPS,
    reason: str,
    config: PipelineConfig
) -> List[int]:
    """
    Populates `selected_source_frames` according to Section 4 of ALGORITHM_SPEC.md:
      - Case 1 (target_fps == 24): Every source frame in [start_frame, end_frame).
      - Case 2 (target_fps == 12): Strictly unique drawing start frames (`run.abs_frame`).
      - Case 3 (target_fps == 8):
          * 3A (`clean_8fps`): Strictly unique drawing start frames (`run.abs_frame`).
          * 3B (`mixed_fallback_8fps`): Subsample K = max(2, round(N / 3)) unique drawings evenly
            if U > K and config.strict_duration_match is True.
    """
    if target_fps == 24:
        return list(range(start_frame, end_frame))

    unique_frames = [r.abs_frame for r in runs]

    if target_fps == 12:
        return unique_frames

    # target_fps == 8
    if reason == "mixed_fallback_8fps" and config.strict_duration_match:
        n_source = end_frame - start_frame
        k_target = max(2, int(round(n_source / 3.0)))
        u_count = len(unique_frames)
        if u_count > k_target:
            indices = np.linspace(0, u_count - 1, k_target, dtype=int)
            # Ensure strictly increasing unique indices
            selected = sorted(list({unique_frames[i] for i in indices}))
            return selected

    return unique_frames


def classify_shot(
    analysis: ShotAnalysis,
    config: PipelineConfig,
    starting_clip_id: int
) -> List[SubClipSpec]:
    """
    Segments a single ShotAnalysis into one or more chronological SubClipSpec objects
    following the 4-step pipeline in ALGORITHM_SPEC.md.

    Args:
        analysis: ShotAnalysis for the current shot.
        config: PipelineConfig instance.
        starting_clip_id: Next global 1-based clip ID counter.

    Returns:
        List of SubClipSpec objects for this shot.
    """
    runs = compute_drawing_runs(analysis)
    if not runs:
        return []

    static_min_frames = max(1, int(round(config.static_hold_sec * 24)))
    min_subclip_frames = max(1, int(round(config.min_subclip_sec * 24)))

    # Step 1: Isolate Static Holds (>= 1.0 Second / 24+ Frames)
    # Partition into (is_static_hold, List[DrawingRun])
    partitions: List[Tuple[bool, List[DrawingRun]]] = []
    current_moving: List[DrawingRun] = []

    for run in runs:
        if run.hold_length >= static_min_frames:
            if current_moving:
                partitions.append((False, current_moving))
                current_moving = []
            partitions.append((True, [run]))
        else:
            current_moving.append(run)

    if current_moving:
        partitions.append((False, current_moving))

    # Build raw subclip tuples: (runs, target_fps, reason)
    raw_subclips: List[Tuple[List[DrawingRun], CadenceFPS, str]] = []
    for is_static, part_runs in partitions:
        if is_static:
            raw_subclips.append((part_runs, 24, "static_hold"))
        else:
            segments = segment_moving_interval(part_runs, min_subclip_frames=min_subclip_frames)
            raw_subclips.extend(segments)

    # Post-Selection Verification:
    # Ensure every 12fps or 8fps subclip has >= 2 unique frames so RIFE always has >= 2 frames to interpolate.
    # If a tiny clip has only 1 unique frame, merge it into the adjacent moving subclip or convert appropriately.
    verified_subclips: List[Tuple[List[DrawingRun], CadenceFPS, str]] = []
    for seg_runs, fps, reason in raw_subclips:
        if fps in (8, 12) and len(seg_runs) < 2:
            if verified_subclips and verified_subclips[-1][2] != "static_hold":
                prev_runs, _, _ = verified_subclips.pop()
                combined_runs = prev_runs + seg_runs
                new_fps, new_reason = classify_single_interval(combined_runs)
                if new_fps in (8, 12) and len(combined_runs) < 2:
                    new_fps, new_reason = 24, "clean_24fps"
                verified_subclips.append((combined_runs, new_fps, new_reason))
            else:
                # Single drawing at start of shot or after static hold that is < static_min_frames
                verified_subclips.append((seg_runs, 24, "clean_24fps"))
        else:
            verified_subclips.append((seg_runs, fps, reason))

    # Construct SubClipSpec objects and enforce Final Zero-Duplicate Safety Assertion
    subclips: List[SubClipSpec] = []
    clip_id = starting_clip_id
    shot_start = analysis.shot.start_frame

    for seg_runs, fps, reason in verified_subclips:
        s_frame = seg_runs[0].abs_frame
        e_frame = seg_runs[-1].abs_frame + seg_runs[-1].hold_length

        selected = select_frames_for_subclip(
            runs=seg_runs,
            start_frame=s_frame,
            end_frame=e_frame,
            target_fps=fps,
            reason=reason,
            config=config,
        )

        # Final Zero-Duplicate Safety Assertion for 12fps and 8fps outputs
        if fps in (8, 12):
            filtered_selected: List[int] = []
            for idx, abs_f in enumerate(selected):
                rel_idx = abs_f - shot_start
                is_new = analysis.is_new_drawing[rel_idx]
                diff_val = analysis.diff_scores[rel_idx]
                if idx == 0:
                    filtered_selected.append(abs_f)
                else:
                    assert is_new and diff_val >= config.diff_threshold, (
                        f"Duplicate frame {abs_f} (diff={diff_val:.3f}) detected in {fps}fps sub-clip!"
                    )
                    filtered_selected.append(abs_f)
            selected = filtered_selected

        subclips.append(
            SubClipSpec(
                clip_id=clip_id,
                parent_shot_index=analysis.shot.shot_index,
                start_frame=s_frame,
                end_frame=e_frame,
                target_fps=fps,
                reason=reason,
                selected_source_frames=selected,
            )
        )
        clip_id += 1

    return subclips


def classify_all_shots(
    analyses: List[ShotAnalysis],
    config: PipelineConfig
) -> List[SubClipSpec]:
    """
    Classifies all analyzed shots in chronological order and assigns global 1-based clip IDs.

    Args:
        analyses: List of ShotAnalysis objects across the entire video.
        config: PipelineConfig instance.

    Returns:
        Chronological list of all SubClipSpec objects.
    """
    all_subclips: List[SubClipSpec] = []
    next_clip_id = 1

    for analysis in analyses:
        shot_subclips = classify_shot(analysis, config=config, starting_clip_id=next_clip_id)
        all_subclips.extend(shot_subclips)
        next_clip_id += len(shot_subclips)

    return all_subclips