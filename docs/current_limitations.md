# Current Limitations

This pipeline is intentionally local and file-based. It does not automate every production step.

## Not Automated Yet

- AI image generation is external and manual through Codex image generation or other tools.
- TTS and voiceover generation are external.
- Transcript and SRT generation are external.
- CapCut audio polish is manual.
- YouTube upload, thumbnail selection, captions, description, pinned comment, and final launch actions may require manual steps.
- Competitor research is not integrated yet.

## Practical Impact

The code is best at deterministic local processing: parsing transcripts, building visual beats, writing prompt records, validating generated images, rendering with MoviePy, and producing advisory audit reports.

The highest-value improvements are likely around reducing manual waiting time, improving script and prompt quality before expensive production steps, and making render performance faster without weakening the existing reliability checks.
