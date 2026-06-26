# Pipeline Overview

This pipeline is a local Python workflow for turning a dark philosophical video essay idea into a rendered YouTube video. It is source-code driven, but several production steps still happen outside the codebase.

## Flow

1. Write `input/script.md` using the templates in `templates/`.
2. Run the script runtime audit to check estimated duration and hard runtime limits.
3. Run the script quality audit to catch structural, hook, payoff, specificity, and retention-risk issues.
4. Generate voiceover externally.
5. Generate transcript or SRT externally.
6. Place local production inputs under `input/`.
7. Run `python -m youtube_pipeline --dry-run` to parse transcript segments, create visual beats, write local metadata, create placeholder images, and build a contact sheet.
8. Run `python -m youtube_pipeline --generate-prompts` to produce `data/image_prompts.json`.
9. Generate images externally, then save one validated PNG per beat in `assets/generated_images/`.
10. Run `python -m youtube_pipeline --validate-generated-images` to verify image count, naming, size, file type, and aspect ratio.
11. Run `python -m youtube_pipeline --use-generated-images` for a static generated-image render.
12. Run `python -m youtube_pipeline --use-generated-images-kinetic` for the kinetic MoviePy render.
13. Run `python -m youtube_pipeline --production-audit` to create advisory readiness reports.
14. Optionally polish audio in CapCut and prepare the YouTube launch package manually.

## Important Boundaries

The repository does not contain private media inputs or outputs. `input/`, `output/`, generated media, prompt records, audit reports, and launch-package artifacts are local working files and are ignored by Git.

The pipeline does not call LLMs, TTS services, image-generation APIs, YouTube APIs, or CapCut. It creates structured local artifacts that make those external steps easier to perform and audit.
