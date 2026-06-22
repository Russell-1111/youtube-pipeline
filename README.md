# YouTube Pipeline

Local-only Python pipeline that turns `input/voiceover.mp3` and `input/transcript.srt` into metadata, placeholder beat images, a contact sheet, and `output/final_video.mp4`.

## Setup

```powershell
python -m pip install -e .[test]
```

## Run

```powershell
python -m youtube_pipeline
```

Dry run skips MP4 rendering:

```powershell
python -m youtube_pipeline --dry-run
```

Use a custom config:

```powershell
python -m youtube_pipeline --config path/to/config.yaml
```

## Outputs

- `data/transcript_segments.json`
- `data/beats.json`
- `data/manifest.csv`
- `assets/images/beat_001.png`, one image per beat
- `assets/contact_sheets/contact_sheet.png`
- `output/final_video.mp4`

V1 is deterministic and local-only. It does not generate AI images, upload to YouTube, create thumbnails, write scripts, use a database, run a dashboard, or call remote services.

## V2 Generated Image Workflow

V2 adds a local Codex-assisted generated-image workflow. The pipeline writes structured prompts, but it does not call image-generation APIs, LLMs, or remote services.

First generate V1 beat data:

```powershell
python -m youtube_pipeline --dry-run
```

Generate deterministic image prompts:

```powershell
python -m youtube_pipeline --generate-prompts
```

This requires `data/beats.json`, writes `data/image_prompts.json`, and creates `assets/generated_images/`. Save one generated PNG per beat using this naming pattern:

```text
assets/generated_images/beat_001.png
assets/generated_images/beat_002.png
assets/generated_images/beat_003.png
```

Validate generated images:

```powershell
python -m youtube_pipeline --validate-generated-images
```

Validation requires every beat image to exist, open as PNG, be at least `1280x720`, and use an exact 16:9 pixel ratio. The preferred size is `1920x1080`; valid non-preferred sizes warn but pass. Extra files in `assets/generated_images/` warn but do not fail.

Render with generated images:

```powershell
python -m youtube_pipeline --use-generated-images
```

This validates first, then renders `output/final_video_generated.mp4`. It never falls back to V1 placeholder frames.
