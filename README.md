# NBA Betting Model

An end-to-end prediction engine that flags mispriced prop lines on **Underdog Fantasy**. It ingests a decade of NBA data, engineers team- and player-level features, trains gradient-boosted models, compares model probabilities against Underdog's implied probabilities, and assembles correlation-aware multi-pick entries sized with fractional Kelly. An LLM narrative layer explains every pick using the model's actual evidence, with a verification step that blocks hallucinated claims.

## How It Works

```
 Data ingestion          Feature engineering        Modeling                 Betting                    Narrative
┌──────────────────┐    ┌─────────────────────┐    ┌───────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│ Basketball Ref.  │    │ Team: ELO, form,    │    │ Game outcome:     │    │ Edge: model prob −  │    │ Evidence packet from │
│ nba_api          │ →  │ pace, Four Factors, │ →  │ XGBoost ensemble  │ →  │ Underdog implied    │ →  │ model outputs        │
│ Underdog lines   │    │ schedule, H2H       │    │                   │    │                     │    │        ↓             │
│ ESPN injuries    │    │ Player: rolling     │    │ Player props:     │    │ Entries: correlation│    │ LLM spiel (Claude)   │
│ PostgreSQL       │    │ stats, usage,       │    │ quantile LightGBM │    │ -aware stacking     │    │        ↓             │
└──────────────────┘    │ matchups, injuries  │    │ per stat          │    │                     │    │ Verifier (no LLM) —  │
                        └─────────────────────┘    └───────────────────┘    │ Sizing: ¼-Kelly     │    │ rejects any claim    │
                                                                            └─────────────────────┘    │ not in the evidence  │
                                                                                                       └──────────────────────┘
```

### 1. Data ingestion (`src/data/`)

- `scrapers/bbref.py` — Basketball Reference box scores and schedules (10 seasons, rate-limited).
- `nba_api_client.py` — official NBA stats: team and player game logs.
- `scrapers/underdog.py` — Underdog Fantasy prop and game lines via the public `over_under_lines` endpoint.
- `scrapers/injury_report.py` — ESPN injury report (OUT / DOUBTFUL statuses drive projection adjustments).
- `db.py` + `migrations/` — PostgreSQL storage for games, logs, lines, and bet tracking.

### 2. Feature engineering (`src/features/`)

All features are computed **as of the day before the game** — no future data ever leaks into a feature (walk-forward everywhere).

Team features: MOV-adjusted ELO with season carryover regression, win streaks, rest days and back-to-backs, head-to-head records, rolling (L5/L10/L20) offensive/defensive/net ratings, pace, and Dean Oliver's Four Factors.

Player features: rolling stat averages (L5/L10/L20/season) for every prop stat, minutes and usage-rate context, home/away splits, historical performance vs. tonight's opponent, and teammate-absence effects.

### 3. Models (`src/models/`)

- **Game outcomes** — an XGBoost classifier ensembled with an ELO-only logistic baseline and a Four Factors LightGBM, with calibrated probabilities (isotonic/sigmoid). Trained walk-forward: 2015–2022 train, 2023 validate, 2024 test.
- **Player props** — one quantile LightGBM per stat (PTS, REB, AST, FG3M, STL, BLK, TOV, plus PRA/PR/PA/RA combos), each predicting a 10th/50th/90th percentile interval rather than a point estimate. The interval is converted to `P(stat > line)` via a normal approximation.
- **Backtesting** (`backtest.py`) — simulated Underdog entries against the real payout structure, reporting win rate, ROI, Sharpe, max drawdown, and calibration.

### 4. Bet selection (`src/betting/`)

- **Edge** (`edge_calculator.py`): `edge = P_model − P_underdog_implied`, in percentage points. Picks below the minimum edge threshold (default 4pp) are discarded.
- **Injury & matchup adjustments** (`scripts/daily_pipeline.py`): projections shift when teammates or opponents are OUT — preferring observed historical "player X sat" deltas, falling back to proportional usage redistribution — and scale with the opponent's defensive profile for each stat.
- **Entries** (`entry_builder.py`): Underdog pays exponentially for multi-pick entries, so correlated picks (same game, same direction, pace stacks, blowout fades) are stacked together using rule-based correlation scores.
- **Sizing** (`kelly.py`): fractional Kelly (quarter-Kelly default) against the multi-pick payout structure.
- **Tracking** (`tracker.py`, `scripts/settle_results.py`, `scripts/pnl_report.py`): every entry logged, settled against actuals from nba_api, and rolled into P&L reports.

### 5. LLM narrative layer (`src/narrative/`)

Every pick gets a short spiel explaining *why the model likes it* — grounded exclusively in the model's own evidence, never in the LLM's outside knowledge.

**Pipeline per pick:**

1. **Evidence packet** (`evidence.py`) — a flat dict of verified facts pulled straight from the pick object and feature parquet: the line, model projection and quantile range, model vs. implied probability, edge, rolling averages, matchup history, usage, and any injury adjustments applied. This packet is the *only* citable material.
2. **Generation** (`generator.py`) — the packet is serialized to JSON and sent to the Anthropic API (`NARRATIVE_MODEL`, default `claude-haiku-4-5`) with hard rules: use only evidence facts, never compute new numbers, never mention players outside the evidence.
3. **Verification** (`verifier.py`) — a deterministic gate with **no LLM involved**, so it cannot itself hallucinate:
   - every number in the text must match an evidence value at the cited precision (or a rolling-window constant like "last 5 games");
   - the direction word must match the pick (an "over" pick may never be described with "under");
   - the pick's player/team must be named, and no other known player may appear unless listed OUT/DOUBTFUL in the evidence.
4. **Retry → fallback** — a failing narrative is regenerated once with the specific violations fed back. If it fails again (or no API key is set), a template narrative assembled directly from the evidence is used instead. The tests enforce that template output always passes verification, so **no unverified claim ever reaches the pick sheet**.

Narratives surface in two places: the daily pick sheet (`--narratives` flag prints a "WHY THESE PICKS" section) and the dashboard (`GET /api/narrative?player=...&stat=...&line=...`).

## Setup

Requires Python 3.11+ and PostgreSQL.

```bash
git clone https://github.com/ticklemepark/nba-betting-model.git
cd nba-betting-model
pip install -e ".[dev]"

cp .env.example .env   # then fill in values
python scripts/setup_db.py
```

`.env` keys:

| Key | Required | Purpose |
|-----|----------|---------|
| `DATABASE_URL` | yes | PostgreSQL connection string |
| `ODDS_API_KEY` | optional | The Odds API market consensus lines |
| `ANTHROPIC_API_KEY` | optional | LLM narratives (template fallback without it) |
| `NARRATIVE_MODEL` | optional | Anthropic model override (default `claude-haiku-4-5`) |

Build features and train models (one-time, then periodically):

```bash
python scripts/build_historical_features.py   # team logs → game features parquet
python scripts/build_player_features.py       # player logs → player features parquet
python scripts/train_game_outcome.py
python scripts/train_player_props.py
```

## Daily Usage

```bash
# Morning: refresh today's features + injury report
python scripts/build_today_game_features.py

# Generate the pick sheet (with narratives)
python scripts/daily_pipeline.py --narratives --bankroll 1000 --min-edge 0.04

# Dry run — no DB writes
python scripts/daily_pipeline.py --dry-run --narratives

# Web dashboard → http://localhost:5000
python scripts/app.py

# After games: settle results and report P&L
python scripts/settle_results.py
python scripts/pnl_report.py
```

Example pick sheet output:

```
Entry 1: 2-pick (payout 3×)  win_prob=38.2%  EV=+0.15  Bet: $18.50
  Nikola Jokic (DEN vs SAS) PRA OVER 52.5  [model=56.8, edge=+7.2%]
  Jamal Murray (DEN vs SAS) PTS OVER 24.5  [model=27.1, edge=+5.1%]

============================================================
  WHY THESE PICKS
============================================================

• Nikola Jokic|PRA|52.5|over
  The model projects 56.8 PRA against a line of 52.5 — a 61.4% chance on the
  OVER side vs Underdog's implied 50%, an 11.4pp edge. He is averaging 55.2
  in his last 5 and 58.1 in 3 matchups against San Antonio this season.
```

## Testing

```bash
python -m pytest        # 438 tests
```

The suite covers scrapers (fixture HTML), every feature module, model train/predict round-trips, edge and Kelly math, entry correlation logic, and the narrative layer (including the verifier's hallucination checks and the template round-trip invariant).

## Repository Layout

```
src/
├── data/        # scrapers, nba_api client, PostgreSQL, migrations
├── features/    # team/ and player/ feature modules + pipeline orchestrator
├── models/      # game outcome, player props, calibration, ensemble, backtest
├── betting/     # edge calculator, Kelly sizing, entry builder, tracker
├── narrative/   # evidence packets, LLM generation, verification, fallbacks
└── utils/       # constants, dates, logging
scripts/         # daily pipeline, training, dashboard, settlement, reports
templates/       # dashboard UI
tests/           # pytest suite
```

## Disclaimer

This is a personal research project. Sports betting involves real financial risk, model edges are estimates that can and do evaporate, and past backtest performance does not guarantee future results. Nothing here is financial advice — bet responsibly and only what you can afford to lose.
