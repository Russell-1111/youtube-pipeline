# YouTube Pipeline

A local Python pipeline for producing dark philosophical YouTube video essays from script/audio/transcript into generated image prompts, validated visual beats, contact sheets, production audits, and kinetic MoviePy renders.

This public repository is the source-code and documentation version of the pipeline. Production assets, generated images, voiceovers, transcripts, rendered videos, thumbnails, captions, and YouTube launch package files are intentionally excluded.

## Setup

```powershell
python -m pip install -e .[test]
```

## Current Workflow

1. Write the script in `input/script.md`.
2. Audit the script runtime with `python -m youtube_pipeline --script-audit input/script.md`.
3. Audit script quality with `python -m youtube_pipeline --script-quality-audit input/script.md`.
4. Generate the voiceover externally.
5. Create the transcript or SRT externally.
6. Place the voiceover and transcript locally under `input/`.
7. Run a dry run to produce metadata, beats, placeholder images, and contact sheets.
8. Generate image prompts from the visual beats.
9. Generate images externally from the prompt records.
10. Validate generated images against the beat manifest.
11. Render the final generated-image video.
12. Render the final kinetic generated-image video.
13. Optionally do a manual CapCut audio polish pass.
14. Prepare the YouTube launch package manually.

The script audits are local heuristic gates. They help catch runtime, structure, and retention-risk issues before voiceover generation, but they are not guarantees of audience performance.

## Main Commands

```powershell
python -m youtube_pipeline --dry-run
python -m youtube_pipeline --script-audit input/script.md
python -m youtube_pipeline --script-quality-audit input/script.md
python -m youtube_pipeline --generate-prompts
python -m youtube_pipeline --validate-generated-images
python -m youtube_pipeline --use-generated-images
python -m youtube_pipeline --use-generated-images-kinetic
python -m youtube_pipeline --production-audit
python -m pytest
```

`--dry-run` reads the local voiceover and transcript, builds transcript segments, visual beats, a manifest, placeholder beat images, and a contact sheet, but skips MP4 rendering.

`--generate-prompts` reads `data/beats.json` and writes structured prompt records to `data/image_prompts.json`. The pipeline does not call an image-generation API.

`--validate-generated-images` checks generated beat images in `assets/generated_images/`. Each beat image must exist, open as PNG, be at least `1280x720`, and use an exact 16:9 ratio. `1920x1080` is preferred.

`--use-generated-images` validates first, then renders `output/final_video_generated.mp4` with generated images.

`--use-generated-images-kinetic` validates first, then renders `output/final_video_generated_kinetic.mp4` with subtle deterministic MoviePy motion.

`--production-audit` writes local advisory production-readiness reports under `data/` without rendering or calling remote services.

## Folder Structure

- `src/youtube_pipeline/` - pipeline source code, CLI entrypoint, parsing, beat generation, prompt generation, generated-image validation, contact sheet creation, audits, and MoviePy rendering.
- `tests/` - pytest coverage for CLI modes, prompt generation, generated-image validation, script audits, production audit, SRT parsing, manifests, and kinetic rendering helpers.
- `templates/` - manual copy-paste templates for script briefs and script-writing prompts.
- `config.yaml` - local default paths, output settings, video dimensions, and beat timing settings.
- `input/` - local private working folder for `script.md`, voiceover files, and transcript/SRT files. Ignored by Git.
- `output/` - local private render output folder. Ignored by Git.
- `assets/images/` - local placeholder beat images. Ignored by Git.
- `assets/generated_images/` - local externally generated image files. Ignored by Git.
- `assets/generated_images_backups/` - local generated-image backups. Ignored by Git.
- `assets/contact_sheets/` - local contact sheet outputs. Ignored by Git.
- `data/` - local generated metadata, beat records, prompt records, manifests, and audit reports. Ignored by Git.
- `docs/` - practical public documentation for future AI-assisted optimization work.

The ignored local folders are required for production runs but are not part of the public repository.

## Current Known Bottlenecks

- External image generation for many visual beats is slow.
- The kinetic render took about 57 minutes for a 9:53 video.
- Render timeout risk exists for long kinetic MoviePy renders.
- Manual CapCut audio polish is currently outside the pipeline.
- Script quality is heuristic, not a virality guarantee.
- Competitor research is not automated yet.
- The current pipeline optimizes reliability over speed.

## Optimization Targets

- Reduce the number of visual beats without reducing retention.
- Generate fewer but stronger prompts.
- Improve the prompt batching workflow.
- Optimize MoviePy render performance.
- Explore FFmpeg-based kinetic rendering alternatives.
- Cache static image transformations.
- Reduce unnecessary per-frame computation.
- Improve the script brief and quality gate using competitor-style references.
- Improve title, thumbnail, description, captions, and launch packaging workflow.

## Public Repo Warning

This repository is intended for code review, documentation, and AI-assisted optimization. It intentionally excludes private production assets, raw media, generated images, voiceover files, transcript files, final videos, thumbnails, captions, CapCut exports, and YouTube launch package files.
