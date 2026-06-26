# Optimization Targets

Future AI assistance should focus on speed, quality, and fewer manual steps without turning the pipeline into an overbuilt platform.

## Image Generation Time

The generated-image stage is external and slow when every visual beat needs a separate image. Useful improvements include reducing beat count, grouping similar beats, improving prompt reuse, and creating better batching instructions for image-generation tools.

## Kinetic Render Time

The kinetic MoviePy render is reliable but slow. A previous 9:53 video took about 57 minutes to render. Optimization should look at reducing per-frame work, caching static image transformations, lowering unnecessary composition cost, and testing FFmpeg-based motion alternatives.

## Script Quality

The script quality audit is a local heuristic gate. It can be improved with stronger brief fields, better section-level scoring, competitor-style references, and clearer warnings about weak hooks, vague claims, missing payoff, or low emotional progression.

## Prompt Quality

Prompt generation should produce fewer, stronger, more batchable prompts. Better prompts should preserve the dark philosophical tone while reducing repeated imagery and avoiding visual beats that are expensive to generate but weak for retention.

## Production Audit Usefulness

The production audit should stay advisory and practical. It should flag missing files, mismatched durations, risky image coverage, overwrite risks, and launch-readiness gaps without blocking unrelated local work or requiring remote services.

## YouTube Launch Packaging

Title, thumbnail, description, captions, pinned comment, and upload checklist work is still mostly manual. The next useful step is a structured local packaging report, not direct YouTube upload automation.

## Avoiding Overengineering

Prefer narrow CLI modes, deterministic reports, and source-controlled templates. Avoid databases, dashboards, background workers, or remote integrations unless they remove a real bottleneck.
