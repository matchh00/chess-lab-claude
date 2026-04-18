# Chess Lab

A modular research platform for studying LLM-driven decision-making in chess.

The lab is not a chess engine. It is a controlled environment for testing a decision framework: extract structured meaning from a position, weight it by policy, compose it into a narrative, present a bounded candidate set, and ask an LLM to choose. Every intermediate artifact is logged so experiments can be compared.

**Core research question:** Does structured narrative context improve LLM move quality versus structured data alone, measured in centipawn loss, controlling for candidate set quality?

---

## How it works

```
board state
    │
    ▼
primitive extraction          30 deterministic primitives (material, king safety,
    │                         development, center, tactics, pawns, initiative)
    ▼
policy weighting              effective_score = normalized_value × weight × confidence
    │                         loaded from YAML — never hardcoded
    ▼
position narrative            template-based prose, max 150 words
    │
    ├── candidate generation  engine top-N + forcing moves + exploratory
    │       │
    │       ├── filter        remove illegal / duplicate moves
    │       ├── rank          internal scoring (hidden from LLM)
    │       ├── shuffle       randomize presentation order (controls positional bias)
    │       └── annotate      primitive deltas, risk flags, candidate narratives
    │
    ▼
LLM prompt (versioned)        position narrative + policy summary + shuffled candidates
    │                         no engine evals shown — narrative-only from v1.1
    ▼
LLM decision                  structured JSON response, validated, fallback chain
    │
    ▼
MoveTrace saved               primitives, weights, narrative, prompt, decision, CPL
```

Internal candidate rank and presentation index are always stored separately. The LLM never sees the engine's ranking order.

---

## Setup

**Requirements:** Python 3.11+, Stockfish

```bash
# Install Stockfish (macOS)
brew install stockfish

# Create and activate virtualenv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install anthropic python-dotenv tiktoken python-chess pydantic pyyaml pandas

# Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

---

## Running experiments

```bash
# Run a batch experiment from a config
python -m src.experiments.run_experiment configs/experiments/baseline.yaml --games 5

# Compare two or more runs side by side
python -m src.experiments.compare_runs baseline_20260418_201653 narrative_off_20260419_090000

# Inspect a game move by move
python -m src.ui.trace_viewer --game_id <game-id>
```

Each run produces:

```
data/runs/{run_id}/
  manifest.json              run config, game IDs, timing
  games/{game_id}.json       full GameTrace per game
  traces/{game_id}.json      MoveTrace per lab move (primitives, prompt, decision)
  move_log.csv               one row per lab move
  game_summary.csv           one row per game
  run_report.json            aggregate metrics
  primitive_attribution.csv  Pearson correlation of each primitive with CPL

reports/latest/{run_id}_summary.md   markdown report with key metrics and attribution table
```

---

## Experiment configs

| Config | Player | Candidates | Narrative | Purpose |
|--------|--------|------------|-----------|---------|
| `baseline.yaml` | LLM | engine-assisted | on | primary condition |
| `narrative_off.yaml` | LLM | engine-assisted | off | narrative ablation |
| `llm_raw_control.yaml` | LLM raw | none | off | pipeline isolation baseline |
| `heuristic_only.yaml` | LLM | heuristic-only | on | no-engine candidate generation |

All configs pit the lab (white) against Stockfish skill 3. Game count, seed, max moves, and prompt version are all configurable per experiment.

---

## Policy profiles

Weights are loaded from YAML at runtime. Available profiles:

| Profile | Character |
|---------|-----------|
| `balanced` | moderate emphasis across all categories |
| `aggressive` | higher weight on initiative, threats, piece activity |
| `defensive` | higher weight on king safety and blunder avoidance |
| `development_first` | prioritizes development and central control |
| `endgame_clean` | prioritizes pawn structure, king activity, simplification |

---

## Prompt versions

| Version | Candidate block |
|---------|----------------|
| `v1.0` | SAN + UCI + engine eval (cp) + source label + risk flags |
| `v1.1` | SAN + UCI + narrative + risk flags only — no engine signals |

v1.1 is the current default. It removes engine evaluations and source labels from the prompt so the LLM cannot shadow the engine's ranking. This is the scientifically honest version for testing narrative value.

---

## Primitive library

30 primitives across 8 categories, each with a normalized score (0–1, higher = better), confidence value, and text render:

| Category | Primitives |
|----------|------------|
| Material | material difference, bishop pair, rook count difference, queen presence |
| Development | minor pieces developed, castled status, undeveloped back-rank minors |
| King safety | pawn shield quality, enemy attackers near king, open lines toward king |
| Center | center occupancy, center attacks, center pawn presence |
| Tactical | hanging pieces (own/opponent), checks available, captures available, threatened majors |
| Piece activity | knight outpost, bishop activity, rook on open file, queen overextension risk |
| Pawn structure | passed pawns, isolated pawns, doubled pawns, backward pawns, pawn islands |
| Initiative | immediate threat, forcing moves count, tempo-gaining candidate |

---

## Project structure

```
src/
  environment/      board manager, engine wrapper, opponents
  primitives/       registry, extractor, 30 primitive definitions
  policies/         YAML profiles, weighting, text summaries
  narratives/       position narrative, candidate narratives, token budgets
  candidates/       generator → filter → rank → shuffle → annotate
  llm/              prompt builder, versioning, schemas, client, decision engine
  gameplay/         move loop, game runner, LLM and raw players
  analytics/        move/game/run metrics, primitive attribution, report builder
  experiments/      run_experiment.py, compare_runs.py
  storage/          Pydantic v2 models for all data objects
  ui/               CLI trace viewer

configs/
  policies/         balanced.yaml, aggressive.yaml, defensive.yaml, ...
  experiments/      baseline.yaml, narrative_off.yaml, llm_raw_control.yaml, ...
  llm/prompts/      v1.0.md, v1.1.md

tests/              192 tests, all passing
```

---

## Current results

| Run | Prompt | Avg CPL | Rank-1 rate | Games |
|-----|--------|---------|-------------|-------|
| baseline (v1.0) | engine evals visible | 232 | 65% | 5 |
| baseline (v1.1) | narrative only | 577 | 41% | 5 |

The rank-1 rate drop (65% → 41%) when engine signals were removed confirms the LLM was shadowing the engine's evaluation numbers in v1.0. The v1.1 CPL establishes the honest floor from which narrative improvements will be measured.

Full results and analysis: [RESEARCH_LOG.md](RESEARCH_LOG.md)

---

## Tech stack

- [`python-chess`](https://python-chess.readthedocs.io) — board logic and legal move generation
- [`pydantic`](https://docs.pydantic.dev) v2 — all data models
- [Stockfish](https://stockfishchess.org) — candidate generation and position evaluation
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — LLM calls (claude-sonnet-4-6)
- [`tiktoken`](https://github.com/openai/tiktoken) — token counting for narrative budgets
- [`pandas`](https://pandas.pydata.org) — analytics and CSV reporting
- YAML — all policy weights and experiment configs (never hardcoded)
