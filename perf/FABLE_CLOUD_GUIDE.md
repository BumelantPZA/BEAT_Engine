# Running the GPU kernel question with Fable (local Code session)

Fable only reads code and writes a plan to `fable/PROPOSALS.md`; it runs and changes nothing.
The implementation and benchmarking happen afterwards in the regular session.

1. In the Claude app's Code tab, start a new session with the folder
   `~/Desktop/Claude/Boundarylab/beat-engine-fable` and the model **Fable**. If there's an
   effort choice, medium is enough. The brief (`CLAUDE.md`) loads automatically.
2. Paste:
   > Read CLAUDE.md and fable/KERNEL_EXCERPT.jl, then write fable/PROPOSALS.md as described
   > (diagnosis, ranked ideas, implementation plan, test plan). Don't run anything and
   > don't change code. Stop when the file is written.
3. If it starts running commands or opening lots of files, stop it and paste: "Stop. Read
   only CLAUDE.md and fable/KERNEL_EXCERPT.jl, then write the plan."
4. When the file exists, tell the regular session "Fable is done". It reads
   `beat-engine-fable/fable/PROPOSALS.md` and implements it in `beat-engine-test`.

(The branch is also on GitHub at BumelantPZA/BEAT_Engine `fable/gpu-kernel`, but that
isn't needed anymore.)
