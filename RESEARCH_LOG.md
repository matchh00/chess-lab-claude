# Chess Lab — Research Log

## Purpose

This log tracks experiment results, design decisions, and open questions across
runs of the chess lab. Each entry records what changed, what was measured, and
what it implies for the next experiment.

The core research question:

> **Does structured narrative context improve LLM move quality versus structured
> data alone, measured in centipawn loss, controlling for candidate set quality?**

---

## Experiment Index

| Run | Config | Prompt | Games | Avg CPL | Rank-1 Rate | Tokens |
|-----|--------|--------|-------|---------|-------------|--------|
| baseline_20260418_201653 | baseline | v1.0 | 5 | 232.2 | 64.6% (93/144) | 119,753 |
| *(planned)* | baseline | v1.1 | 5 | ~577 | ~41% | — |
| *(planned)* | narrative_off | v1.1 | 20 | — | — | — |
| *(planned)* | llm_raw_control | v1.1 | 20 | — | — | — |
| *(planned)* | heuristic_only | v1.1 | 20 | — | — | — |

---

## Entry 001 — 2026-04-18

### Prompt v1.0 → v1.1: Engine signals removed

**Change:** Candidate block was restructured in v1.1 to strip all engine signals:
centipawn evaluations, source labels (`engine` / `heuristic` / `random`), and
internal rank were removed. The LLM now sees only SAN, a prose candidate
narrative, and risk flags per candidate.

**Rationale:** The v1.0 prompt was leaking strong engine priors into the
presentation layer. A candidate listed as `engine | eval: +41.0 cp` is not a
meaningful test of whether narrative improves decisions — the LLM could ignore
the narrative entirely and just follow the highest eval. Removing this signal
forces the LLM to reason from position context, policy, and narrative alone.

**Format change (v1.0 → v1.1):**

```
# v1.0 (engine signals visible)
- [5] e4 (uci: e2e4) | eval: +41.0 cp | Engine slightly favors this move.

# v1.1 (narrative only)
[5] e4 (e2e4) — Controls the center and opens diagonals for active piece play. Risk: none.
```

### Key finding: rank-1 selection dropped, CPL increased

When centipawn numbers were removed (v1.0 → v1.1):

- **Rank-1 selection rate dropped from 65% to 41%.**
- **Average centipawn loss increased from 232 to 577.**

This is the expected direction — and the reason v1.1 is more scientifically
honest. In v1.0, the LLM was largely shadowing the engine's top pick, using the
eval number as a proxy for quality. In v1.1 it is making independent judgments,
which are noisier but more meaningful as a test of the narrative system.

The 65% rank-1 rate in v1.0 is partially an artifact of the prompt leaking
engine rankings. The true baseline behavior — what the LLM does with narrative
alone — is captured by v1.1.

The elevated CPL in v1.1 is the cost of removing the training wheels. It
establishes the floor from which narrative improvements will be measured.

### Baseline run details (v1.0, 5 games)

| Metric | Value |
|--------|-------|
| Games | 5 |
| Wins / Draws / Losses | 0 / 0 / 2 (3 hit move limit) |
| Avg CPL | 232.2 |
| Total blunders | 7 |
| Total token usage | 119,753 |
| Avg tokens / move | ~831 |
| Rank-1 selected | 93 / 144 moves (64.6%) |

**Presentation index distribution** (positional bias check):

| Index | Count |
|-------|-------|
| 0 | 25 |
| 1 | 21 |
| 2 | 20 |
| 3 | 19 |
| 4 | 22 |
| 5 | 21 |
| 6 | 6 |
| 7 | 10 |

Distribution is roughly flat for indices 0–5, confirming the shuffler is doing
its job — no strong positional bias toward first or last candidate.

**Primitive attribution (top 10 by |correlation| with CPL):**

| Primitive | Correlation with CPL | N |
|-----------|---------------------|---|
| self_material_difference | −0.249 | 144 |
| self_rook_count_difference | −0.207 | 144 |
| self_isolated_pawns | −0.193 | 144 |
| self_backward_pawns | −0.183 | 144 |
| self_bishop_activity | −0.173 | 144 |
| self_minor_pieces_developed | −0.163 | 144 |
| self_forcing_moves_count | −0.149 | 144 |
| self_bishop_pair | −0.146 | 144 |
| self_pawn_islands | −0.142 | 144 |
| self_immediate_threat | −0.134 | 144 |

Negative correlations mean higher primitive scores (better position on that
dimension) corresponded to lower CPL. This is the expected direction. The
material and pawn structure primitives show the strongest signal.

`self_castled_status` and `self_undeveloped_back_rank_minors` both show
**positive** correlation with CPL — meaning positions where we had not yet
castled (or had more undeveloped pieces) were associated with higher CPL. This
aligns with chess intuition but also suggests those primitives may need higher
weights in the policy to influence LLM decisions more strongly.

---

## Known Issues

### Knight outpost and immediate threat primitives (v1.1)

In v1.1 runs, `self_knight_outpost` and `self_immediate_threat` appear to be
leading the LLM astray. Candidate narratives that mention knight outpost
opportunities or immediate threats cause the LLM to prefer those moves even
when the tactical logic is shallow (e.g. a non-forcing "threat" that is easily
parried). Two hypotheses:

1. The narrative templates for these primitives overstate certainty — the
   text render reads as more decisive than the underlying primitive supports.
2. The LLM pattern-matches on the word "threat" and weights it too heavily
   relative to structural signals like king safety or development.

**Short-term fix candidates:**
- Lower confidence on `self_knight_outpost` (currently 1.0 — consider 0.7)
- Revise the render template for `self_immediate_threat` to use more hedged
  language ("may create a threat" vs. "creates an immediate threat")
- Add a `self_threat_quality` primitive that distinguishes forcing from
  non-forcing threats

These changes will require a new primitive library version bump and a re-run
to measure impact.

### `self_queen_overextension_risk` returns NaN in attribution

The `queen_overextension_risk` primitive has `confidence=0.6` and appears to
have zero variance across many positions (it returns 0.0 whenever the queen is
not visibly overextended). This leads to a NaN Pearson correlation. The
primitive is not currently providing useful attribution signal. Options:
- Refine the extraction heuristic to produce more variance
- Accept that it's low-signal and leave it

---

## Planned Experiments

### narrative_off (next)

**Config:** `configs/experiments/narrative_off.yaml`  
**Purpose:** Direct test of the core research question. Same pipeline as
baseline v1.1 but with position narrative suppressed. LLM receives structured
candidate data and policy text only — no prose position narrative.

**Hypothesis:** CPL will be higher than narrative_on under v1.1, confirming
that narrative context provides real decision quality improvement beyond
structured data alone.

**Control variable:** candidate set is identical (engine_assisted), so any CPL
difference is attributable to the narrative, not candidate quality.

### llm_raw_control (next)

**Config:** `configs/experiments/llm_raw_control.yaml`  
**Purpose:** Baseline isolation. The LLM receives only the FEN and legal move
list — no primitives, no narrative, no candidates, no policy. This measures
raw LLM chess ability without any pipeline support.

**Hypothesis:** CPL will be substantially higher than both narrative_on and
narrative_off, confirming the pipeline adds value independent of the narrative
layer.

**Use:** Sets the floor. If pipeline CPL ≈ llm_raw CPL, the whole pipeline is
not helping. If pipeline CPL is substantially lower, we have evidence the
structured context is doing real work.

### heuristic_only

**Config:** `configs/experiments/heuristic_only.yaml`  
**Purpose:** Tests whether removing the engine from candidate generation degrades
decision quality. Candidates are generated by primitive-aligned heuristics only
(MVV-LVA captures, checks, central pawns, castling), not engine top-N.

**Hypothesis:** CPL will be higher than engine_assisted because the candidate
pool is weaker. The magnitude of the difference tells us how much the engine's
candidate guidance contributes to performance, independent of the LLM's
reasoning quality.

**Secondary question:** Does the LLM do better or worse at selecting rank-1
candidates when engine signals are absent from both the candidate pool and the
prompt?

---

## Next Workstream — UI

A browser-based or richer terminal UI is planned for the next workstream.
Priority features:

1. **Run comparison dashboard** — side-by-side CPL, win rate, rank distribution,
   and primitive attribution across multiple `run_id`s. Currently available via
   `python -m src.experiments.compare_runs` in the terminal.

2. **Primitive library browser** — view all 30 primitives with their current
   weights, confidence values, categories, and example renders. Useful for
   diagnosing attribution issues like the knight outpost problem above.

3. **Narrative explorer** — for a saved game, show the position narrative and
   candidate narratives side by side with the board state. Helps identify
   whether narrative text is accurate and decision-relevant.

4. **Attribution timeline** — plot how each primitive's correlation with CPL
   evolves across games within a run. Early games may show different patterns
   than late games as positions become more complex.

The CLI trace viewer (`python -m src.ui.trace_viewer`) handles move-by-move
inspection for now. The richer UI will build on the same data layer
(`data/runs/{run_id}/`).

---

## Open Questions

1. Does the narrative system improve decisions, or is the candidate set quality
   the dominant variable? (answered by narrative_off experiment)

2. Does the LLM exhibit positional bias (preference for first or last candidate)?
   (current data: no strong bias in v1.0; need v1.1 data to confirm)

3. Which primitives are most causally related to low CPL, vs. merely correlated?
   (attribution gives correlation; causal direction requires weight ablation)

4. Is 5 games enough to draw conclusions? (likely not — 20-game runs planned)

5. Should primitive weights be updated between runs based on attribution results?
   (Phase 8 weight adaptation; not yet implemented)
