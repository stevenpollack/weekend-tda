# AGENTS.md

This repository's agent guidance lives in [`CLAUDE.md`](CLAUDE.md) — read it in full.

It is kept as a single canonical file so the two can't drift. The short version:

- This is a **time-boxed weekend calibration exercise**, not a research project. Source of truth for scope, phases, and stop conditions is [`plans/weekend_tda_brief.md`](plans/weekend_tda_brief.md).
- **Question:** can persistent homology detect the topological signature of one semantic class removed from a dataset before embedding?
- **Fixed stack, no substitutions:** CLIP ViT-B/32 · CIFAR-100 · giotto-tda. One model, one dataset, one TDA library.
- **Hard stop Sunday 6pm.** If time runs out, ship what exists.
- **Don't oversell.** A clean negative beats a noisy positive; the most likely outcome is no detectable signal, and that is a valid finding.
- Respect the inter-phase **decision points** — if the toy sanity check fails, pivot to learning rather than pushing forward.

See `CLAUDE.md` for the full phase table, compute limits, and deliverable spec.
