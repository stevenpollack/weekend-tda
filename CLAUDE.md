# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A **time-boxed calibration exercise**, not a research project. The question:

> If we artificially remove one semantic class from a dataset before embedding it,
> can persistent homology detect a meaningful topological signature of that absence?

Full brief: [`plans/weekend_tda_brief.md`](plans/weekend_tda_brief.md). Read it before doing anything substantive — it is the source of truth for scope, phases, and stop conditions.

The goal is a **calibrated opinion** on whether the broader idea (empty regions in latent space encoding meaningful absences) is worth pursuing — not a polished result. Ambiguous or negative results are the expected, valuable outcome.

## Core constraints (do not violate)

- **Hard stop: Sunday 6pm.** ~14 working hours total over the weekend.
- **One model:** CLIP ViT-B/32 (`open_clip`, `pretrained='openai'`). No other encoders.
- **One dataset:** CIFAR-100 test set. No other datasets.
- **One TDA library:** giotto-tda. No other persistent-homology libraries.
- **No paper-reading during the weekend** — that's post-weekend (brief §10).
- **Don't generalize** from one experiment, one model, one dataset.

## The four phases

| Phase | Goal | Key artifact |
|-------|------|--------------|
| 1. Synthetic sanity check | Verify pipeline + build PD intuition on toy data | notebook cells |
| 2. Real embeddings baseline | CLIP + CIFAR-100 embeddings, UMAP, baseline persistence diagrams | `embeddings.npy`, `labels.npy`, `umap_full.png` |
| 3. Ablation | Remove a class, compare diagrams, bottleneck distance, localize hole | comparison figures |
| 4. Writeup | One-page honest summary | `RESULTS.md` |

There are **decision points** between phases (brief §4.4, §6.5). Respect them — if Phase 1 toy ablation doesn't produce the expected clean result, the brief says stop and pivot to reading tutorials, not push forward.

## Working style here

- **Don't oversell.** A clean negative is more valuable than a noisy positive. Report what the diagrams actually show, including "looks identical / can't attribute the difference."
- **Most likely outcome (~70%) is no detectable signal** from a 1% perturbation. That's a finding, not a failure. If it happens, the brief's prescribed next step is the more aggressive ablation (remove a full 5-class superclass), not endless tweaking.
- **Run both 512-dim and PCA-50 variants** in Phases 2–3. The disagreement between them is itself the signal worth reporting.
- **Watch compute.** Vietoris–Rips on full 10k×512 is infeasible. Subsample (stratified, ~2000 pts; drop to 1000 if a single computation exceeds ~30 min). Never wait hours on one cell.
- **Save embeddings to disk** (`.npy`) so they're never recomputed.
- **Commit at checkpoints** (end of each phase). First commit already includes the brief.

## Stop / escalate conditions

End at the **first** of: Sunday 6pm, Phase 4 complete, or 3+ hours sunk on a single technical issue (dependency hell, slow compute, mysterious bug). If time runs out mid-project, **ship what exists** — a writeup covering only Phases 1–2 plus tooling lessons is a valid deliverable.

## Environment

Managed with **uv**. Python pinned to **3.12** (`.python-version`); `requires-python = ">=3.12,<3.13"`. The `<3.13` ceiling is deliberate — giotto-tda has no 3.13 wheels yet, and without the ceiling uv silently backslides to the unbuildable `giotto-tda==0.1.4`. Stay on 3.12 until giotto-tda ships 3.13 wheels.

- Reproduce env: `uv sync`. Add a dep: `uv add <pkg>` (updates `uv.lock`). Run anything: `uv run <cmd>` (e.g. `uv run jupyter lab`).
- Core deps (in `pyproject.toml`): `torch torchvision open_clip_torch giotto-tda umap-learn matplotlib scikit-learn`. Dev group: `jupyter ipykernel`.
- Commit `pyproject.toml`, `uv.lock`, `.python-version`; ignore `.venv/` and data artifacts (see `.gitignore`).
- **Positron:** select `.venv/bin/python` as the interpreter (Python: Select Interpreter); the notebook kernel follows from `ipykernel` in `.venv`. No manual `activate` needed inside the IDE.
- Primary surface is a Jupyter notebook (`pilot.ipynb`). CPU is fine; CUDA used if available (torch is the `+cu130` build).

## Deliverable

`RESULTS.md`, one page max, four sections: what I ran / what I found (with 2–3 figures) / what didn't work or surprised me / should this be pursued further (calibrated yes-no-conditional with one reason). Then push to GitHub.
