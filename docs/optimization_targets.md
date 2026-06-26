# Optimization Targets

Future AI assistance should focus on speed, quality, and fewer manual steps without turning the pipeline into an overbuilt platform.

## Image Density And Generation Time

The generated-image stage is external and slow when every visual beat needs a separate image. Useful improvements include better prompt variety, stronger batching instructions for image-generation tools, and render infrastructure that can handle more images without lowering quality.

The current render optimization goal is not to reduce image density. Future Late Human videos should support roughly 70-90 high-quality generated images across 9-11 minutes while keeping render time practical.

## Kinetic Render Time

The kinetic MoviePy render is reliable but slow. A previous 9:53 video took about 57 minutes to render. The MoviePy renderer should remain the trusted fallback while an experimental FFmpeg renderer is tested as a separate command.

Initial render optimization should prioritize benchmark reporting, sample renders under `output/benchmarks/`, metadata validation, and an FFmpeg per-beat segment plus concat architecture. This is safer on Windows than one large monolithic filter graph, and it leaves room for future beat-level caching without implementing caching immediately.

Minimum speed target: 2x faster than the MoviePy baseline. Strong target: 3x or better if visual quality remains equivalent. If benchmarks show those targets are unrealistic, document the measured evidence rather than claiming equivalence.

## Script Quality

The script quality audit is a local heuristic gate. It can be improved with stronger brief fields, better section-level scoring, competitor-style references, and clearer warnings about weak hooks, vague claims, missing payoff, or low emotional progression.

## Prompt Quality

Prompt generation should produce stronger, more varied, more batchable prompts. Better prompts should preserve the dark philosophical tone while reducing repeated imagery and avoiding visual beats that are expensive to generate but weak for retention.

## Production Audit Usefulness

The production audit should stay advisory and practical. It should flag missing files, mismatched durations, risky image coverage, overwrite risks, and launch-readiness gaps without blocking unrelated local work or requiring remote services.

## YouTube Launch Packaging

Title, thumbnail, description, captions, pinned comment, and upload checklist work is still mostly manual. The next useful step is a structured local packaging report, not direct YouTube upload automation.

## Avoiding Overengineering

Prefer narrow CLI modes, deterministic reports, and source-controlled templates. Avoid databases, dashboards, background workers, or remote integrations unless they remove a real bottleneck.
