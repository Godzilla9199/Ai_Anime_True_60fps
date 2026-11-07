# AI Anime & Video True 60fps — Cadence Splitter & Deduplicator

An automated Python CLI pre-processing pipeline designed for **AI Frame Interpolation (RIFE 4.7+)**. 

Standard AI frame interpolators assume uniform frame pacing and fail when given variable-cadence video—such as traditional animation (animated on 1s, 2s, and 3s), stuttery gameplay recordings, or low-FPS generative AI video—resulting in severe "stutter-step" judder and morphing artifacts.

This pipeline ingests a video (`.mkv` / `.mp4`), detects hard scene cuts, analyzes per-frame motion holds, routes segments into native **`08fps/`**, **`12fps/`**, and **`24fps/`** buckets, and strips duplicate frames so AI interpolation models can smoothly upscale every clip to true 60fps+.

## Why This Matters for the AI Ecosystem
- **Zero-Judder Animation Interpolation:** Separates 8fps (Threes), 12fps (Twos), and 24fps (Ones) so RIFE interpolates true motion vectors instead of duplicate held frames.
- **Compute-Efficient Generative AI Video:** Allows generative video models to render at lower native frame rates (8fps/12fps) to save local GPU compute, then cleanly pre-processes them for 60fps interpolation.
- **Stuttery Gameplay & Capture Recovery:** Deduplicates dropped/repeated frames in screen recordings before optical-flow or AI frame interpolation.

## Key Features
- **Automated Scene Detection:** Uses `PySceneDetect` (`ContentDetector`) to ensure hard cuts are never blended across clips.
- **Compression-Resilient Frame Diffing:** Uses downscaled `3x3` Gaussian-blurred grayscale Mean Absolute Difference (MAD) combined with the 98th percentile pixel difference (`max(mad, p98 * 0.25)`) to catch both full-screen pans and small character mouth/eye flaps while ignoring H.264/HEVC macroblock noise.
- **Static Hold Isolation:** Automatically isolates `>= 1.0s` (`24+` frames) static pauses as locked `24fps` clips so interpolators do not morph across intentional holds.
- **Zero-Duplicate 12fps & 8fps Exports:** Drops held duplicate frames and encodes native `8fps`, `12fps`, and `24fps` MP4 clips in a single sequential pass using `libx264 -crf 12 -pix_fmt yuv420p`.
- **Chronological Manifests:** Outputs `Clip_0001_24fps.mp4`, `Clip_0002_12fps.mp4`, etc., alongside `timeline_manifest.json` and `timeline_manifest.csv` for automated reassembly.

## Installation
1. Ensure **Python 3.10+** and **FFmpeg** (`ffmpeg` on system `PATH`) are installed.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt