# YouTube Pipeline V1

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
