"""
Manifest writer and Rich terminal summary generator.
Exports `timeline_manifest.json` and `timeline_manifest.csv` into the output directory.
"""

import csv
import json
import os
from typing import Any, Dict, List, Tuple
from rich.console import Console
from rich.table import Table

from src.config import PipelineConfig, SubClipSpec, VideoProperties

console = Console()


def write_manifests(
    subclips: List[SubClipSpec],
    video_props: VideoProperties,
    config: PipelineConfig
) -> Tuple[str, str]:
    """
    Writes `timeline_manifest.json` and `timeline_manifest.csv` containing detailed metadata
    for every generated clip.

    Args:
        subclips: List of exported SubClipSpec objects.
        video_props: Source video properties.
        config: PipelineConfig instance.

    Returns:
        Tuple of (json_manifest_path, csv_manifest_path).
    """
    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, "timeline_manifest.json")
    csv_path = os.path.join(config.output_dir, "timeline_manifest.csv")

    source_fps = video_props.fps if video_props.fps > 0 else 24.0

    clip_records: List[Dict[str, Any]] = []
    for clip in subclips:
        source_frame_count = clip.end_frame - clip.start_frame
        exported_frame_count = len(clip.selected_source_frames)
        source_duration_sec = round(source_frame_count / source_fps, 4)
        exported_duration_sec = round(exported_frame_count / float(clip.target_fps), 4)
        rel_path = f"{clip.folder_name}/{clip.filename}"

        clip_records.append(
            {
                "clip_id": clip.clip_id,
                "filename": clip.filename,
                "relative_path": rel_path,
                "parent_shot_index": clip.parent_shot_index,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "source_frame_count": source_frame_count,
                "exported_frame_count": exported_frame_count,
                "target_fps": clip.target_fps,
                "reason": clip.reason,
                "source_duration_sec": source_duration_sec,
                "exported_duration_sec": exported_duration_sec,
                "selected_source_frames": clip.selected_source_frames,
            }
        )

    manifest_data = {
        "source_video": {
            "input_path": os.path.abspath(config.input_path),
            "width": video_props.width,
            "height": video_props.height,
            "source_fps": video_props.fps,
            "total_frames": video_props.total_frames,
        },
        "config": {
            "diff_threshold": config.diff_threshold,
            "scene_threshold": config.scene_threshold,
            "min_subclip_sec": config.min_subclip_sec,
            "static_hold_sec": config.static_hold_sec,
            "crf": config.crf,
            "strict_duration_match": config.strict_duration_match,
        },
        "summary": {
            "total_clips": len(subclips),
            "clips_08fps": sum(1 for c in subclips if c.target_fps == 8),
            "clips_12fps": sum(1 for c in subclips if c.target_fps == 12),
            "clips_24fps": sum(1 for c in subclips if c.target_fps == 24),
        },
        "clips": clip_records,
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(manifest_data, jf, indent=2)

    csv_fields = [
        "clip_id",
        "filename",
        "relative_path",
        "parent_shot_index",
        "start_frame",
        "end_frame",
        "source_frame_count",
        "exported_frame_count",
        "target_fps",
        "reason",
        "source_duration_sec",
        "exported_duration_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        for rec in clip_records:
            row = {k: rec[k] for k in csv_fields}
            writer.writerow(row)

    return json_path, csv_path


def print_summary_table(subclips: List[SubClipSpec], output_dir: str) -> None:
    """Prints a formatted Rich summary table of all exported clips and folder counts."""
    bucket_table = Table(title="Anime Cadence Splitter - FPS Bucket Summary")
    bucket_table.add_column("Folder", style="bold cyan")
    bucket_table.add_column("Target FPS", justify="right")
    bucket_table.add_column("Clip Count", justify="right", style="bold green")
    bucket_table.add_column("Total Exported Frames", justify="right")

    for fps in (8, 12, 24):
        matching = [c for c in subclips if c.target_fps == fps]
        frames_sum = sum(len(c.selected_source_frames) for c in matching)
        bucket_table.add_row(
            f"{fps:02d}fps/",
            f"{fps} fps",
            str(len(matching)),
            str(frames_sum),
        )

    console.print()
    console.print(bucket_table)

    clip_table = Table(title="Chronological Sub-Clip Breakdown")
    clip_table.add_column("Clip ID", style="cyan", justify="right")
    clip_table.add_column("Filename", style="bold white")
    clip_table.add_column("Shot #", justify="right")
    clip_table.add_column("Source Range", justify="center")
    clip_table.add_column("Target FPS", justify="right", style="green")
    clip_table.add_column("Exported Frames", justify="right")
    clip_table.add_column("Classification Reason", style="yellow")

    for c in subclips[:50]:  # Cap display at 50 rows to keep terminal clean
        clip_table.add_row(
            f"{c.clip_id:04d}",
            f"{c.folder_name}/{c.filename}",
            str(c.parent_shot_index),
            f"[{c.start_frame}, {c.end_frame})",
            f"{c.target_fps} fps",
            str(len(c.selected_source_frames)),
            c.reason,
        )

    console.print()
    console.print(clip_table)
    if len(subclips) > 50:
        console.print(f"[dim]... and {len(subclips) - 50} more clips (see timeline_manifest.json).[/dim]")
    console.print(f"\n[bold green]Done![/bold green] Output saved to: [underline]{os.path.abspath(output_dir)}[/underline]\n")