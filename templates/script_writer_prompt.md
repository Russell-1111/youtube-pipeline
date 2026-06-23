You are writing a dark productivity / time / modern life philosophical essay for a YouTube voiceover.

The script must be written to pass:
1. python -m youtube_pipeline --script-audit input/script.md
2. python -m youtube_pipeline --script-quality-audit input/script.md

Do not finalize the script unless it includes SCRIPT_BRIEF and VOICEOVER_START / VOICEOVER_END markers.

Target runtime:
- Aim for 10:00-11:00.
- Keep the script under the 11:30 hard cap.
- Use minimal pause tags. Only include pause tags for major beats.

Structure:
- Use exactly five spoken sections.
- Section 1: Hook / opening tension. The first 30 seconds must justify the click immediately.
- Section 2: Setup / problem frame. Frame the viewer's problem without slow throat-clearing.
- Section 3: Core idea / explanation. Explain the mechanism behind the tension.
- Section 4: Deeper turn / consequence. Complicate the idea or reveal the hidden cost.
- Section 5: Payoff / ending. Resolve or reframe the opening loop with a strong final idea.

Required SCRIPT_BRIEF fields:
- title
- thumbnail_promise
- viewer_problem
- central_tension
- central_question
- viewer_payoff
- emotional_arc
- open_loop
- final_takeaway
- target_runtime
- target_viewer

Style:
- Dark productivity / time / modern life philosophical essay.
- Reflective but not slow.
- Cinematic but not bloated.
- Poetic but not vague.
- Use self-improvement psychology only when it clarifies the idea.
- Avoid generic motivational slogans.
- Avoid "in today's video", "we're going to talk about", "let's dive in", "at the end of the day", "in conclusion", "nowadays", "many people", "society today", and "we all know".

Retention requirements:
- The hook must connect clearly to the title and thumbnail promise.
- The opening must create tension, contradiction, consequence, or a high-stakes question.
- The script must maintain an open loop.
- Each section must move the idea forward.
- Include meaningful turns using contrast, escalation, or payoff language.
- The ending must answer or reframe the opening tension, not merely summarize.

Output format:

<!-- SCRIPT_BRIEF
title:
thumbnail_promise:
viewer_problem:
central_tension:
central_question:
viewer_payoff:
emotional_arc:
open_loop:
final_takeaway:
target_runtime: 10:00-11:00
target_viewer:
-->

<!-- VOICEOVER_START -->

## 1. Hook / opening tension

## 2. Setup / problem frame

## 3. Core idea / explanation

## 4. Deeper turn / consequence

## 5. Payoff / ending

<!-- VOICEOVER_END -->
