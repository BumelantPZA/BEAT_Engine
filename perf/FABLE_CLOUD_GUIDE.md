# Running a planning question with Fable (local Code session)

Fable only reads code and writes a plan; it runs and changes nothing. The implementation and
benchmarking happen afterwards in the regular session.

Round 2 (whole-solve plan, 2026-09-26). The brief is `perf/FABLE_PLAN_HANDOFF.md`, copied to
`beat-engine-fable/CLAUDE.md`, and the code excerpt is `beat-engine-fable/fable/SOLVE_EXCERPT.jl`.
Round 1 (GPU kernel) files are in `beat-engine-fable/fable/round1/`.

1. In the Claude app's Code tab, start a new session with the folder
   `~/Desktop/Claude/Boundarylab/beat-engine-fable` and the model **Fable**, effort **high**
   (round 1 used medium). The brief (`CLAUDE.md`) loads automatically.
2. Paste:
   > Follow CLAUDE.md: read fable/SOLVE_EXCERPT.jl, then write fable/PLAN.md. Don't run
   > anything and don't change code. Stop when the file is written.
3. If it runs commands, reads more than ~4 extra files or starts rewriting the plan, stop it and
   paste: "Stop. Use what you have and write fable/PLAN.md once."
4. When the file exists, tell the regular session "Fable is done".

Token budget: round 1 was 4 API calls, ~30k output (mostly thinking), ~225k cache-read and ~69k
cache-write input. Round 2 changes: a 2x larger excerpt (70 KB vs 35 KB), up to 4 targeted extra
reads, effort high instead of medium, and a plan of up to ~20 KB. Expected ~1.5–2x round 1.
