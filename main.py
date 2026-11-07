"""
CLI Entry point for the Anime Cadence Splitter & Deduplicator.
"""

import argparse
import sys
from typing import List, Optional
from rich.console import Console

from src.config import PipelineConfig, SubClipSpec
from src.scene_detector import detect_scenes
from src.frame_analyzer import analyze_shots
from src.cadence_classifier import classify_all_shots
from src.video_exporter import ensure_ffmpeg_available, export_subclips
from src.manifest_writer import write_manifests, print_summary_table

console = Console()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments according to STYLE_AND_RULES.md."""
    parser = argparse.ArgumentParser(
        description="Anime Cadence Splitter & Deduplicator (for RIFE 4.7 60fps Interpolation)"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input .mkv or .mp4 anime video file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output root folder where 08fps/, 12fps/, and 24fps/ will be created (default: ./output).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.2,
        help="Frame difference threshold for detecting duplicate frames (default: 1.2).",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=27.0,
        help="PySceneDetect ContentDetector threshold for hard cuts (default: 27.0).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum sub-clip duration in seconds (default: 0.5s = 12 frames @ 24fps).",
    )
    parser.add_argument(
        "--static-duration",
        type=float,
        default=1.0,
        help="Minimum static pause duration in seconds to keep at 24fps (default: 1.0s = 24 frames).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=12,
        help="H.264 CRF value (0 = lossless, 12 = near-lossless default).",
    )
    parser.add_argument(
        "--keep-all-unique",
        action="store_true",
        default=False,
        help="If set, mixed 2s/3s fallback clips keep all unique drawings instead of subsampling to exact 8fps duration.",
    )
    return parser.parse_args(argv)


def run_pipeline(config: PipelineConfig, show_progress: bool = True) -> List[SubClipSpec]:
    """
    Executes the complete 5-stage Anime Cadence Splitter & Deduplicator pipeline.

    Args:
        config: PipelineConfig settings.
        show_progress: Whether to render terminal progress bars and summary tables.

    Returns:
        List of generated SubClipSpec objects.
    """
    ensure_ffmpeg_available()

    if show_progress:
        console.print(f"[bold blue]Stage 1/5:[/bold blue] Detecting scene cuts in [cyan]{config.input_path}[/cyan]...")
    shots, video_props = detect_scenes(
        video_path=config.input_path,
        threshold=config.scene_threshold,
        show_progress=show_progress,
    )
    if show_progress:
        console.print(
            f"  Detected [bold]{len(shots)}[/bold] shot(s) across [bold]{video_props.total_frames}[/bold] frames "
            f"({video_props.width}x{video_props.height} @ {video_props.fps:.2f} fps)."
        )

    if show_progress:
        console.print("[bold blue]Stage 2/5:[/bold blue] Computing per-frame visual differences...")
    analyses = analyze_shots(
        video_path=config.input_path,
        shots=shots,
        config=config,
        show_progress=show_progress,
    )

    if show_progress:
        console.print("[bold blue]Stage 3/5:[/bold blue] Classifying cadences & deduplicating held frames...")
    subclips = classify_all_shots(analyses=analyses, config=config)

    if show_progress:
        console.print(f"[bold blue]Stage 4/5:[/bold blue] Exporting [bold]{len(subclips)}[/bold] sub-clip(s) via FFmpeg (H.264 CRF {config.crf})...")
    export_subclips(
        video_path=config.input_path,
        subclips=subclips,
        video_props=video_props,
        config=config,
        show_progress=show_progress,
    )

    if show_progress:
        console.print("[bold blue]Stage 5/5:[/bold blue] Writing JSON and CSV timeline manifests...")
    write_manifests(subclips=subclips, video_props=video_props, config=config)

    if show_progress:
        print_summary_table(subclips=subclips, output_dir=config.output_dir)

    return subclips


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    args = parse_args(argv)
    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output,
        diff_threshold=args.threshold,
        scene_threshold=args.scene_threshold,
        min_subclip_sec=args.min_duration,
        static_hold_sec=args.static_duration,
        crf=args.crf,
        strict_duration_match=not args.keep_all_unique,
    )

    try:
        run_pipeline(config, show_progress=True)
        return 0
    except Exception as exc:
        console.print(f"[bold red]Pipeline Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())