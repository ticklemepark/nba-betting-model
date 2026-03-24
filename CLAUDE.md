# CLAUDE.md — NBA Betting Model (Underdog Fantasy)
ssss
## Mission

Build a profitable NBA betting model targeting **Underdog Fantasy** (the only legal platform for our California-based user). The system produces two bet types:

1. **Game Outcomes** — pick winners/losers of NBA games (Underdog's "Rival" picks)
2. **Player Props** — over/under on individual player stat lines (Underdog's "Pick'em" higher/lower)

Profitability is the only metric that matters. We are not building a research toy. Every architectural decision, feature, and model choice must justify itself through expected value (EV).

---

## Platform Constraints: Underdog Fantasy

Underdog is a daily fantasy / pick'em platform, NOT a traditional sportsbook. This fundamentally shapes our strategy:

- **No moneyline / spread / totals** in the traditional sense. Underdog uses a "Pick'em" format where you build entries of 2-6 picks (higher/lower on props, or rival picks on game winners).
- **Payouts scale with number of picks in an entry**: 2-pick = 3x, 3-pick = 6x, 4-pick = 10x, 5-pick = 20x, 6-pick = 36x. This means **correlation between picks matters enormously** — we want to stack correlated picks within entries.
- **No vig in the traditional sense** — Underdog's edge comes from setting lines that are slightly off from true probability. Our edge comes from finding where their lines diverge most from reality.
- **Lines move less frequently** than traditional books, creating stale line opportunities especially around injury news and late scratches.
- **Props available**: Points, Rebounds, Assists, PRA (Points+Rebounds+Assists), P+A, P+R, R+A, Steals, Blocks, Turnovers, 3-Pointers Made, Fantasy Score, and occasionally minutes and double-doubles.
- **CRITICAL**: Underdog lines are often set using simpler models than sharp books. This is our primary edge — we are competing against a softer market.

### Correlation Stacking Strategy

This is the single biggest edge available on Underdog. Because entries require multiple picks and pay exponentially, we should:

- **Stack game totals**: If we predict a high-scoring game, go OVER on points for players on BOTH teams.
- **Stack pace**: Fast-paced matchups inflate all counting stats. Go OVER on PRA for starters in both teams.
- **Inverse correlation**: If a team's star is OUT, go OVER on the backup's props AND UNDER on the opposing team's defensive stats that key off that position.
- **Blowout fading**: If we predict a blowout, go UNDER on the favorite's starters (they sit Q4) and UNDER on the losing team's efficiency stats.
- **Back-to-back exploitation**: Team on a B2B? UNDER on their starters' minutes and counting stats. Their opponents' props go OVER.

---

## System Architecture

```
nba-betting-model/
├── CLAUDE.md                    # This file
├── pyproject.toml               # Project config, dependencies
├── .env                         # API keys, DB credentials (NEVER commit)
├── .env.example                 # Template for .env
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                    # Data ingestion & storage
│   │   ├── __init__.py
│   │   ├── db.py                # PostgreSQL connection manager
│   │   ├── scrapers/
│   │   │   ├── __init__.py
│   │   │   ├── bbref.py         # Basketball Reference scraper (box scores, schedule)
│   │   │   ├── underdog.py      # Underdog Fantasy lines scraper/API
│   │   │   ├── injury_report.py # NBA official injury reports
│   │   │   └── odds_api.py      # The Odds API for market consensus lines
│   │   ├── nba_api_client.py    # nba_api wrapper for official stats
│   │   └── migrations/          # SQL schema migrations
│   │       └── 001_base_tables.sql
│   │
│   ├── features/                # Feature engineering pipeline
│   │   ├── __init__.py
│   │   ├── team/
│   │   │   ├── __init__.py
│   │   │   ├── elo.py           # ELO rating system (EXISTING — see below)
│   │   │   ├── form.py          # Win streaks, recent performance windows
│   │   │   ├── pace.py          # Pace / possessions per game
│   │   │   ├── four_factors.py  # Dean Oliver's Four Factors
│   │   │   ├── schedule.py      # B2B, rest days, travel distance
│   │   │   ├── h2h.py           # Head-to-head season record
│   │   │   └── ratings.py       # Offensive/Defensive ratings, net rating
│   │   ├── player/
│   │   │   ├── __init__.py
│   │   │   ├── rolling_stats.py # Rolling averages (5g, 10g, 20g, season)
│   │   │   ├── matchup.py       # Player vs. position / vs. specific defender
│   │   │   ├── usage.py         # Usage rate, minutes context, role changes
│   │   │   ├── home_away.py     # Home/away splits
│   │   │   └── availability.py  # Injury impact, teammate absence effects
│   │   ├── game/
│   │   │   ├── __init__.py
│   │   │   ├── vegas_implied.py # Implied totals from market consensus
│   │   │   └── matchup_context.py # Combined team+player context for a game
│   │   └── pipeline.py          # Orchestrates feature generation
│   │
│   ├── models/                  # Model training & inference
│   │   ├── __init__.py
│   │   ├── game_outcome.py      # Game winner prediction model
│   │   ├── player_props.py      # Player prop prediction model
│   │   ├── calibration.py       # Probability calibration (Platt scaling)
│   │   ├── ensemble.py          # Model ensembling logic
│   │   └── backtest.py          # Walk-forward backtesting engine
│   │
│   ├── betting/                 # Bet selection & bankroll
│   │   ├── __init__.py
│   │   ├── edge_calculator.py   # Compare model prob vs. implied prob
│   │   ├── kelly.py             # Fractional Kelly criterion for sizing
│   │   ├── entry_builder.py     # Build optimal Underdog entries (correlation-aware)
│   │   └── tracker.py           # Track bets, P&L, ROI
│   │
│   └── utils/
│       ├── __init__.py
│       ├── constants.py         # Team abbreviations, mappings
│       ├── dates.py             # Date/timezone helpers
│       └── logging.py           # Structured logging
│
├── notebooks/                   # Exploration only — production code lives in src/
│   └── feature-engineering.ipynb  # Timothy's original ELO notebook (reference)
│
├── tests/
│   ├── __init__.py
│   ├── test_elo.py
│   ├── test_form.py
│   ├── test_schedule.py
│   ├── test_h2h.py
│   ├── test_bbref.py
│   └── ...
│
├── scripts/
│   ├── daily_pipeline.sh        # Cron: pull data → features → predict → output picks
│   └── backtest_runner.py       # Full historical backtest
│
└── data/
    ├── raw/                     # Scraped CSVs, JSON dumps
    ├── processed/               # Feature matrices, ready for modeling
    └── models/                  # Serialized model artifacts (.joblib)
```

---

## Existing Code: Feature Engineering Notebook

Timothy has a working notebook (`notebooks/feature-engineering.ipynb`) that implements several features. This is the foundation — we are **refactoring it into production modules**, not rewriting from scratch. Here is what it contains and where each piece maps:

### Database Layer
- PostgreSQL connection to a local `nba` database with tables: `traditional_stats_{year}`, `team_stats_{year}`, `games_{year}` for years 2015-2024.
- Team abbreviation mapping (30 NBA teams → standard 3-letter codes).
- **Maps to**: `src/data/db.py` and `src/utils/constants.py`

### Box Score Scraper
- Scrapes Basketball Reference for game URLs by month, then fetches team-level box scores from each game page.
- Handles 10 seasons of data with `time.sleep(3)` rate limiting.
- Produces a DataFrame with columns: `AWAY_MP, AWAY_FG, AWAY_FGA, AWAY_FG%, AWAY_3P, AWAY_3PA, AWAY_3P%, AWAY_FT, AWAY_FTA, AWAY_FT%, AWAY_ORB, AWAY_DRB, AWAY_TRB, AWAY_AST, AWAY_STL, AWAY_BLK, AWAY_TO, AWAY_PF, AWAY_PTS, HOME_MP, HOME_FG, HOME_FGA, HOME_FG%, HOME_3P, HOME_3PA, HOME_3P%, HOME_FT, HOME_FTA, HOME_FT%, HOME_ORB, HOME_DRB, HOME_TRB, HOME_AST, HOME_STL, HOME_BLK, HOME_TO, HOME_PF, HOME_PTS, AWAY, HOME, DATE, SEASON`
- **Maps to**: `src/data/scrapers/bbref.py`

### ELO Rating System (`src/features/team/elo.py`)
The core algorithm already implemented:
```python
# Win probability from ELO ratings (logistic curve with home court advantage)
def win_probs(home_elo, away_elo, home_court_advantage):
    h = 10 ** (home_elo / 400)
    r = 10 ** (away_elo / 400)
    a = 10 ** (home_court_advantage / 400)
    denom = r + a * h
    return (a * h / denom), (r / denom)

# Dynamic K-factor adjusted by margin of victory
def elo_k(MOV, elo_diff):
    k = 20
    if MOV > 0:
        multiplier = (MOV + 3) ** 0.8 / (7.5 + 0.006 * elo_diff)
    else:
        multiplier = (-MOV + 3) ** 0.8 / (7.5 + 0.006 * (-elo_diff))
    return k * multiplier

# Update ELO after a game result
def update_elo(home_score, away_score, home_elo, away_elo, home_court_advantage):
    home_prob, away_prob = win_probs(home_elo, away_elo, home_court_advantage)
    home_win = 1 if home_score > away_score else 0
    away_win = 1 - home_win
    k = elo_k(home_score - away_score, home_elo - away_elo)
    return home_elo + k * (home_win - home_prob), away_elo + k * (away_win - away_prob)

# Season carryover: 75% previous ELO + 25% regression to 1505 mean
def get_prev_elo(team, date, season, box_scores, elo_df):
    # ... looks up most recent ELO for team
    if prev_game_season != season:
        return 0.75 * elo_rating + 0.25 * 1505
    return elo_rating
```

Key design decisions in existing ELO:
- **Home court advantage = 69 ELO points** (roughly 60% implied win probability for equal-ELO teams at home)
- **MOV-adjusted K-factor**: Blowouts move ELO more than close games. The `(MOV+3)^0.8` exponent dampens extreme blowouts.
- **Season regression**: 75/25 blend toward 1505 prevents stale ratings from prior seasons. The 1505 mean (not 1500) slightly rewards teams that made it to the end of the prior season.
- **Starting ELO = 1500** for teams with no history in the dataset.

### Win Streak Tracker (`src/features/team/form.py`)
- Tracks consecutive wins entering each game, reset at season boundaries.
- Produces `HOME_STREAK` and `AWAY_STREAK` columns.

### Back-to-Back Detector (`src/features/team/schedule.py`)
- Flags whether home/away team is playing on consecutive days.
- Handles month/year boundary edge cases.
- Produces `HOME_B2B` and `AWAY_B2B` (binary 0/1).

### Head-to-Head Record (`src/features/team/h2h.py`)
- Tracks within-season win rate of each team against the specific opponent.
- Produces `HOME_REC` and `AWAY_REC` (float 0.0-1.0).

### Current Feature Vector (from notebook)
The notebook merges all features into this shape for game outcome prediction:
```
HOME | AWAY | HOME_AFTER (elo) | AWAY_AFTER (elo) | HOME_STREAK | AWAY_STREAK | HOME_B2B | AWAY_B2B | HOME_REC | AWAY_REC | LABEL (1=home win)
```

---

## Feature Engineering: Full Feature Set

### TEAM-LEVEL FEATURES (Game Outcome Model)

These are all computed **as of the day before the game** to prevent leakage.

| # | Feature | Source | Rationale |
|---|---------|--------|-----------|
| 1 | `home_elo` / `away_elo` | Existing notebook | Captures long-run team strength with MOV adjustment |
| 2 | `elo_diff` | Derived | Single strongest predictor — difference between team ELOs |
| 3 | `home_win_prob_elo` | `win_probs()` | ELO-implied probability, useful as a calibrated baseline |
| 4 | `home_streak` / `away_streak` | Existing notebook | Momentum proxy — teams on hot streaks play with confidence |
| 5 | `home_b2b` / `away_b2b` | Existing notebook | Fatigue signal — B2B teams score ~2-3 fewer PPG historically |
| 6 | `home_rest_days` / `away_rest_days` | Schedule | Expand B2B to continuous: 1 day, 2 days, 3+ days rest |
| 7 | `home_h2h_record` / `away_h2h_record` | Existing notebook | Some teams just match up poorly against others |
| 8 | `home_off_rating_L10` / `away_off_rating_L10` | nba_api | Offensive rating (points per 100 possessions) over last 10 games. Captures current form better than season-long averages. |
| 9 | `home_def_rating_L10` / `away_def_rating_L10` | nba_api | Defensive rating over last 10 games |
| 10 | `home_net_rating_L10` / `away_net_rating_L10` | Derived | Off rating - Def rating. Net rating is the single best simple predictor of team quality. |
| 11 | `home_pace` / `away_pace` | nba_api | Possessions per 48 min. High pace = more possessions = more variance = more opportunities for props. |
| 12 | `projected_pace` | Derived | Average of both teams' pace — predicts game tempo. |
| 13 | `home_efg_pct_L10` / `away_efg_pct_L10` | nba_api / bbref | Effective FG% — weights 3s appropriately. One of Dean Oliver's Four Factors. |
| 14 | `home_tov_pct_L10` / `away_tov_pct_L10` | nba_api | Turnover rate — turnovers kill possessions. |
| 15 | `home_orb_pct_L10` / `away_orb_pct_L10` | nba_api | Offensive rebound rate — second chances extend possessions. |
| 16 | `home_ft_rate_L10` / `away_ft_rate_L10` | nba_api | Free throw rate (FTA/FGA) — getting to the line is a quality indicator. |
| 17 | `home_travel_distance` | Derived from schedule | Miles traveled from last game location. Cross-country trips hurt performance. |
| 18 | `home_days_on_road` | Derived from schedule | Consecutive days away from home city. Road fatigue compounds. |
| 19 | `home_wins_L10` / `away_wins_L10` | Schedule | Simple recent win count — complements ELO (which is slow-moving). |
| 20 | `home_ats_L10` / `away_ats_L10` | Odds API | Against-the-spread record — captures whether team is over/underperforming expectations. |

### PLAYER-LEVEL FEATURES (Player Prop Model)

These target individual stat predictions: points, rebounds, assists, PRA, 3PM, etc.

| # | Feature | Source | Rationale |
|---|---------|--------|-----------|
| 1 | `player_stat_avg_L5` | nba_api game logs | Most recent form. 5-game window captures hot/cold streaks. |
| 2 | `player_stat_avg_L10` | nba_api game logs | More stable recent average. |
| 3 | `player_stat_avg_L20` | nba_api game logs | Medium-term baseline. |
| 4 | `player_stat_avg_season` | nba_api | Full season baseline — smooths out noise. |
| 5 | `player_minutes_avg_L5` | nba_api | Minutes context is everything. A player averaging 35 min who's been playing 28 min last 5 is in a different regime. |
| 6 | `player_usage_rate_L5` | nba_api | How much of the team's offense runs through this player. Usage rate predicts volume stats (points, FGA). |
| 7 | `player_stat_home_avg` / `player_stat_away_avg` | nba_api splits | Some players have massive home/away splits (especially for 3PM and points). |
| 8 | `player_stat_vs_team_avg` | nba_api matchup | Historical performance against this specific opponent. Small sample but sometimes reveals real matchup edges. |
| 9 | `opp_def_rating_vs_position` | nba_api | How many points/rebounds/assists does the opponent allow to this player's position? A center facing the league's worst interior D should go OVER. |
| 10 | `opp_pace` | nba_api | Fast opponents = more possessions = inflated counting stats. |
| 11 | `projected_game_total` | Derived / Odds API | Implied total points for the game. High totals inflate everyone's stats. |
| 12 | `player_team_implied_total` | Derived | What's the team's implied point total? Player should get their usage-rate share of this. |
| 13 | `teammate_out_boost` | Injury report | **KEY EDGE**: When a high-usage teammate is OUT, remaining players absorb their usage, shots, and assists. Quantify the delta. |
| 14 | `player_b2b_flag` | Schedule | Players on B2B often get minutes-managed or just play worse. |
| 15 | `player_rest_days` | Schedule | More rest = fresher legs = better performance, especially for older players. |
| 16 | `player_minutes_projection` | Model | Project minutes first, then project stats per minute. Separates the two sources of variance. |
| 17 | `player_recent_foul_trouble` | Game logs | Players averaging 4+ fouls in recent games might get pulled earlier — UNDER on minutes and stats. |
| 18 | `blowout_risk` | Game outcome model | If our model gives one team >70% win probability, starters on both sides may sit in Q4. UNDER bias. |
| 19 | `player_stat_std_L10` | Derived | Volatility of the stat. High-variance players are harder to predict — factor this into confidence. |
| 20 | `line_vs_projection_delta` | Model output | The actual edge: (our projected stat) - (Underdog's line). Only bet when this exceeds our threshold. |

---

## Modeling Strategy

### Game Outcome Model

**Architecture: Gradient Boosted Trees (XGBoost / LightGBM)**

Why:
- Handles mixed feature types (continuous ELO, binary B2B, percentages) natively.
- Captures non-linear interactions (e.g., B2B matters MORE for older teams).
- Interpretable via SHAP — we can explain WHY we're picking a game.
- Proven in sports modeling — outperforms logistic regression on structured tabular data.

Training:
- Walk-forward validation: train on seasons 2015-2022, validate on 2023, test on 2024. **NEVER use future data.** This is the most common source of false confidence in sports models.
- Target: binary classification (home win = 1, away win = 0).
- Output: calibrated probabilities (apply Platt scaling or isotonic regression post-hoc).
- Evaluate on: log loss, Brier score, and most importantly, **simulated ROI** against historical Underdog lines.

Ensemble approach:
- Model A: XGBoost on full feature set
- Model B: Logistic regression on ELO features only (strong baseline, hard to beat)
- Model C: LightGBM on Four Factors + pace features only
- Final prediction: weighted average of all three, weights tuned on validation set.

### Player Prop Model

**Architecture: Per-stat regression models**

Each stat category (PTS, REB, AST, 3PM, PRA, etc.) gets its own model because the features that matter differ:
- Points → usage rate, team implied total, opponent defensive rating
- Rebounds → player height/position, opponent ORB/DRB rates, game pace
- Assists → player assist rate, teammate quality, game pace
- 3PM → player 3PA volume, 3P%, opponent 3P defense

Training:
- Target: continuous stat value (not over/under — we predict the number, THEN compare to the line).
- Model: LightGBM regression, outputting predicted stat + prediction interval.
- Use quantile regression to get 10th/90th percentile estimates — this directly tells us confidence on over/under.

**Critical**: For props, the model doesn't need to be accurate in absolute terms. It needs to be **better than Underdog's line**. We only bet where our prediction disagrees with theirs by more than our threshold.

### Bet Selection: Edge & Kelly Criterion

```python
# Edge calculation
model_probability = 0.58  # Our model says 58% chance of OVER
implied_probability = 0.50  # Underdog's line implies 50/50 (standard for pick'em)
edge = model_probability - implied_probability  # 0.08 = 8% edge

# Fractional Kelly for position sizing
# Underdog payouts: 2-pick = 3x, 3-pick = 6x, etc.
# For a single pick in a 2-pick entry at 3x payout:
kelly_fraction = (edge * payout - (1 - model_probability)) / payout
bet_size = bankroll * kelly_fraction * 0.25  # Use quarter-Kelly for safety
```

**Minimum edge threshold**: Do NOT bet on anything with less than 4% edge. The variance in NBA is high — small edges get eaten by entropy. Target picks with 6%+ edge for core plays.

### Entry Construction (Underdog-Specific)

This is where we differentiate from generic models. Underdog requires multi-pick entries:

1. **Generate all picks** with positive edge for the slate.
2. **Score correlation** between every pair of picks: positively correlated picks (same game, same direction) amplify each other in multi-pick entries.
3. **Build entries using correlation-aware optimization**:
   - Group picks by game.
   - Prefer entries where all picks are correlated (e.g., all OVERs in a projected high-scoring game).
   - Avoid entries mixing uncorrelated picks from different games — they reduce the probability of sweeping the entry.
4. **Allocate bankroll** across entries using modified Kelly, accounting for the multi-pick payout structure.

---

## Data Pipeline: Daily Workflow

```
────────────────────────────────────────────────────────────────────
  6:00 AM PT — Data Pull
  • Scrape yesterday's box scores (bbref + nba_api)
  • Update ELO ratings for completed games
  • Pull today's injury reports
  • Pull Underdog lines for today's games
  • Pull consensus odds from The Odds API
────────────────────────────────────────────────────────────────────
  6:30 AM PT — Feature Engineering
  • Recompute all rolling windows (L5, L10, L20)
  • Recompute rest days, B2B flags
  • Generate player minutes projections
  • Build feature matrices for today's games + props
────────────────────────────────────────────────────────────────────
  7:00 AM PT — Model Inference
  • Run game outcome model → win probabilities
  • Run player prop models → stat projections + intervals
  • Calculate edge vs. Underdog lines
  • Filter to picks with edge >= threshold
────────────────────────────────────────────────────────────────────
  7:30 AM PT — Entry Building
  • Score correlations between positive-edge picks
  • Build optimal entries (2-pick through 5-pick)
  • Apply Kelly sizing to each entry
  • Output final picks + entries to dashboard/Slack/CLI
────────────────────────────────────────────────────────────────────
  1:00 PM PT — Line Check (lines can move)
  • Re-scrape Underdog lines
  • Recompute edges with updated lines
  • Flag any late injury news that changes projections
  • Update picks if edges shifted significantly
────────────────────────────────────────────────────────────────────
  Post-Games — Results
  • Pull game results and player stats
  • Score all picks (hit/miss) and entries (sweep/bust)
  • Update P&L tracker
  • Log model accuracy metrics
────────────────────────────────────────────────────────────────────
```

---

## Key Edges to Exploit

These are specific, actionable strategies where we believe Underdog's lines are systematically beatable:

### 1. Injury-Driven Usage Redistribution (Highest Confidence Edge)
When a high-usage player is ruled out (especially late — after Underdog sets lines), the remaining players absorb their production. Underdog is slow to adjust prop lines for teammates. Build a "usage redistribution model" that predicts how stats flow when Player X is out.

### 2. Back-to-Back Fatigue
NBA teams on B2Bs score ~2-3 fewer PPG and play worse defensively. This is well-known but Underdog's lines don't always fully price it in, especially for specific player minutes. UNDER on starters' props for B2B teams.

### 3. Pace Mismatch Exploitation
When a top-5 pace team faces a bottom-5 pace team, game tempo is uncertain — but the average tends to favor the faster team more than Underdog implies. Go OVER on the fast team's players' counting stats.

### 4. Rest Advantage
Teams with 2+ days rest vs. teams on 1 day rest have a historically significant edge (~4% win rate bump). This stacks with ELO to produce high-confidence game outcome picks.

### 5. Fourth Quarter Blowout Bias
Underdog props are based on full-game performance. But in blowouts, starters play ~30 min instead of ~36. Our blowout probability model can systematically fade starters in lopsided matchups.

### 6. Three-Point Shooting Variance
3PM is the highest-variance prop. The market tends to set lines near a player's average, but 3PM follows more of a Poisson distribution. Players who shoot high volume 3s at moderate percentages have frequent "boom" games. We can find OVER value when the line is set conservatively.

### 7. Stale Lines After News
Underdog is slower than sharp books to move lines after news drops (lineup changes, rest days announced, etc.). If we automate injury report monitoring, we can identify stale lines within minutes of an announcement.

---

## Technical Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Ecosystem (pandas, sklearn, nba_api), Timothy's existing code |
| Database | PostgreSQL | Already in use, relational fits schedule/game/player data well |
| ML Framework | LightGBM + XGBoost | Industry standard for tabular data, fast training |
| Data | nba_api, Basketball Reference, The Odds API | Comprehensive coverage: official stats, box scores, market lines |
| Scheduling | cron + Python scripts | Simple, reliable. No need for Airflow at this scale. |
| Tracking | PostgreSQL table | Track every bet placed, result, P&L |
| Notifications | Slack webhook or CLI output | Daily pick sheet delivery |

### Dependencies (to be approved before adding)

When proposing a new dependency, always state:
1. What it does
2. Why an existing dep can't handle it
3. Bundle size / maintenance status

Core deps (pre-approved):
- `pandas`, `numpy` — data manipulation (already in use)
- `scikit-learn` — preprocessing, calibration, evaluation
- `lightgbm` — primary model training
- `xgboost` — ensemble member
- `nba_api` — official NBA stats (no alternative for this data)
- `psycopg2` — PostgreSQL driver (already in use)
- `beautifulsoup4`, `requests` — web scraping (already in use)
- `python-dotenv` — env var management
- `joblib` — model serialization
- `shap` — model interpretability (explain picks)

Needs approval before adding:
- `playwright` / `selenium` — only if Underdog requires JS rendering
- Any sportsbook-related API clients — evaluate per-provider

---

## Engineering Rules

### Change Management

1. **For tasks that touch core business logic, change data models, or could affect multiple components**: describe your approach first, list files you'll modify, and wait for approval. For isolated, well-scoped changes (fixing a typo, adding a CSS class, implementing a function with a clear spec): just do it.

2. **If a task requires changes to more than 3 files**, stop and present a numbered task breakdown. Each sub-task should be independently testable. Wait for approval on the plan before starting sub-task 1. Complete each sub-task fully (including verification) before moving to the next.

3. **After completing a feature or making a significant change**, write a brief "risk assessment" comment:
   - (1) What assumption does this code make that could be wrong?
   - (2) What's the most likely way this fails in production?
   - (3) Write at least one test for the happy path and one for the most likely failure mode. Don't list hypothetical edge cases — focus on the failure that would actually happen to a real user.

### Bug Fixing

4. **When there's a bug**, start by writing a failing test that reproduces it. Run the test to confirm it fails for the right reason. Then fix the code until the test passes. Then run the full test suite to check for regressions. Do not skip any of these steps.

### Code Hygiene

5. **Never delete code you don't understand.** If you encounter code that seems unnecessary but you're not sure why it exists, ask before removing it. Legacy code often encodes business logic that isn't obvious from reading it.

6. **When importing a new dependency or library**, state: (1) what it does, (2) why an existing dependency can't handle it, and (3) its bundle size / maintenance status. Get approval before adding to pyproject.toml.

### Data Integrity (Sports-Model-Specific)

7. **NEVER use future data in features.** Every feature must be computed using ONLY data available before the game starts. This is the #1 source of false confidence in sports models. When in doubt, add an assertion: `assert feature_date < game_date`.

8. **Always use walk-forward validation.** Never use random train/test splits for time-series sports data. Train on past seasons, validate on the next season, test on the most recent season.

9. **Treat scraped data as untrusted.** Basketball Reference can have typos, missing games, or format changes. Validate row counts, check for null scores, and alert on anomalies.

---

## Backtesting Protocol

Backtesting is how we know if we're profitable BEFORE risking money. Do it rigorously:

1. **Walk-forward only**: Train on data up to date T, predict games on date T+1. Never look ahead.
2. **Simulate actual Underdog entries**: Don't just measure single-pick accuracy. Build simulated 2-5 pick entries and track simulated P&L against the actual payout structure.
3. **Track by bet type**: Separate metrics for game outcomes vs. each prop category. You'll find some prop types are more profitable than others.
4. **Minimum sample**: Don't trust results on fewer than 200 bets. NBA has ~1,230 regular season games and ~8,000+ prop opportunities per season. Backtest across at minimum 2 full seasons.
5. **Report these metrics**:
   - Win rate (overall and by bet type)
   - ROI (return on investment: profit / total wagered)
   - CLV (closing line value: did our picks beat where the line closed?)
   - Sharpe ratio of daily P&L
   - Maximum drawdown (worst peak-to-trough loss)
   - Calibration plot (predicted probability vs. actual win rate)

---

## Profitability Targets

Based on Underdog's payout structure, here's what we need:

| Entry Type | Payout | Break-even Win Rate | Target Win Rate |
|-----------|--------|--------------------:|----------------:|
| 2-pick | 3x | 33.3% | 38%+ |
| 3-pick | 6x | 16.7% | 20%+ |
| 4-pick | 10x | 10.0% | 13%+ |
| 5-pick | 20x | 5.0% | 7%+ |

For single-pick accuracy (before combining into entries):
- Game outcomes: target **58%+ accuracy** (home win base rate is ~57%, so we need to beat that on our filtered picks)
- Player props: target **55%+ accuracy** on bets we choose to take (we only bet where edge exceeds threshold, so accuracy on selected bets should be higher than overall accuracy)

---     dddd

## Development Phases

### Phase 1: Data Foundation (Week 1-2) — STATUS: COMPLETE
- [x] Refactor notebook scraper into `src/data/scrapers/bbref.py`
- [x] Set up PostgreSQL schema with migrations (`src/data/migrations/001_base_tables.sql`)
- [x] Refactor ELO into `src/features/team/elo.py` with tests
- [x] Refactor win streak, B2B, H2H into their respective modules with tests
- [x] `src/data/db.py` — PostgreSQL connection manager
- [x] `src/utils/constants.py` — team abbreviations and cross-reference maps
- [x] **Verified**: `scripts/verify_phase1.py` confirms ELO + streaks are exact matches on 1230 2023-24 games (via nba_api); B2B matches exactly (no expansion teams in 2024); H2H diffs are pre-documented notebook data leakage, not our bugs. 10/10 checks pass.

### Phase 2: Feature Expansion (Week 3-4) — STATUS: COMPLETE
- [x] `src/data/nba_api_client.py` — nba_api wrapper (team + player game logs)
- [x] `src/data/migrations/002_team_game_logs.sql` — DB table for raw team logs
- [x] `src/features/team/ratings.py` — rolling off/def/net ratings (L5/L10/L20)
- [x] `src/features/team/pace.py` — rolling pace + projected game pace (L5/L10/L20)
- [x] `src/features/team/four_factors.py` — rolling eFG%, TOV%, ORB%, FTR (L5/L10/L20)
- [x] `src/features/team/schedule.py` — added `compute_rest_days` (continuous 1–7)
- [x] `src/features/team/form.py` — added `compute_wins_rolling` (wins in last N games)
- [x] Player-level feature pipeline (`src/features/player/`) — rolling_stats, usage, home_away, matchup, availability
- [x] Injury report data source (`src/data/scrapers/injury_report.py`) — ESPN scraper
- [x] Feature pipeline orchestrator (`src/features/pipeline.py`) — `build_game_features` + `build_player_features`
- [x] **Verified**: 187/187 tests passing (all team + player + pipeline modules covered)

### Phase 3: Modeling (Week 5-6) — STATUS: COMPLETE
- [x] `scripts/build_historical_features.py` — fetch nba_api team logs + run pipeline, save to parquet
- [x] `src/models/calibration.py` — manual isotonic/sigmoid calibration (sklearn 1.8+ compatible)
- [x] `src/models/game_outcome.py` — XGBoost classifier, walk-forward train/eval, save/load
- [x] `src/models/ensemble.py` — EloLogitModel + FourFactorLGBModel + EnsembleModel (weight tuning)
- [x] `src/models/backtest.py` — BacktestResult: win rate, ROI, Sharpe, max drawdown, calibration curve
- [x] `src/models/player_props.py` — PlayerPropModel (3 quantile LightGBM), train_all_props, run_prop_backtest
- [x] `scripts/build_player_features.py` — fetch player + team logs (20 API calls), save player_features.parquet
- [x] `scripts/train_player_props.py` — train all stat models, run backtests, print summary, save artifacts
- [x] **Verified**: 283/283 tests passing

### Phase 4: Underdog Integration (Week 7-8) — STATUS: COMPLETE
- [x] `src/data/migrations/003_betting_tables.sql` — underdog_lines, bet_entries, entry_picks tables
- [x] `src/data/scrapers/underdog.py` — unofficial Underdog API client (prop + game lines, `UNDERDOG_TOKEN` in .env)
- [x] `src/betting/edge_calculator.py` — `PropPick`, `GamePick`, `screen_prop_picks`, `screen_game_picks`
- [x] `src/betting/kelly.py` — `fractional_kelly`, `size_entry`, `summarise_sizing`
- [x] `src/betting/entry_builder.py` — `score_correlation`, `build_entries`, `rank_entries`
- [x] `src/betting/tracker.py` — `log_entry`, `settle_entry`, `get_pnl_summary`
- [x] `scripts/daily_pipeline.py` — full daily orchestration (fetch → screen → build → size → log)
- [x] **Verified**: 369/369 tests passing

### Phase 5: Go Live (Week 9+) — STATUS: INFRASTRUCTURE COMPLETE
- [x] `src/data/scrapers/underdog.py` — switched to public `/beta/v6/over_under_lines` endpoint (no auth, no token expiry)
- [x] `scripts/app.py` — Flask web dashboard with `/api/props`, `/api/player-stats`, `/api/rankings`, `/api/status`
- [x] `templates/index.html` — dark-theme prop dashboard UI with sorting, filtering, edge badges, player drill-down panel
- [x] `scripts/settle_results.py` — post-game settlement: fetches actuals from nba_api, marks entries won/lost
- [x] `scripts/pnl_report.py` — P&L summary report (overall + by entry size + daily breakdown + Phase 3 comparison)
- [x] **Verified**: 393/393 tests passing
- [ ] Run daily pipeline in shadow mode (paper trading) for 2 weeks
- [ ] Compare paper results to backtest expectations
- [ ] If within expected range — go live with quarter-Kelly sizing
- [ ] Gradually increase sizing as track record builds confidence

### Phase 6: Dashboard Intelligence (Week 10+) — STATUS: COMPLETE
The dashboard is now a full model UI, not just a prop browser.

**Edge unification**:
- Dashboard and pipeline now use the **same edge formula**: `P(stat > line) [ML model] − P(over) [Underdog implied probability]`, in percentage-points (pp).
- Edge displays as `+7.2pp` (model) or `+4.8%*` (rolling average fallback when models not loaded).
- Model predictions load **eagerly** from `data/models/*.joblib` + `data/processed/player_features.parquet` on first request — no "Analyze All" needed for model edge.

**O/U odds fix**:
- `american_price` from the Underdog public API is stored as an implied probability.  The Underdog app shows **entry payout multipliers** (0.85×, 1.06×); these are a different representation.
- Dashboard now only renders the O/U Odds column when the line is **asymmetric** (`|over_payout − under_payout| > 0.01`). Symmetric standard lines show `–`.

**Model features in player panel** (`/api/player-stats`):
- `feat_l5`, `feat_l10`, `feat_season` — rolling stat averages from the model's parquet (vs NBA API live values)
- `home_avg`, `away_avg`, `vs_opp_avg` — home/away and matchup splits
- `usage_l5`, `usage_l10` — usage rate proxy from player features
- `teammate_boost` — usage redistribution when a high-usage teammate is out
- `feature_date` — date of the most recent feature row (freshness indicator)

**Double-doubles / triple-doubles** (computed from existing NBA API game logs, no new calls):
- `dd_pts_reb_L10`, `dd_pts_ast_L10`, `triple_double_L10` shown in player panel

**New dashboard columns**: Model (projection median + P(over)%), Edge (model pp / rolling %*)
**New dashboard filters**: "Model edge only", "Edge ≥ 4", "Edge ≥ 6"
**Game log additions**: W/L result, FG%, +/- columns

**TODO (Phase 6 remaining)**:
- [ ] Q1/Q2 per-game period stats — requires `BoxScoreByPeriodV2` per game (20+ extra API calls per player). Feasible but expensive. Implement as opt-in endpoint.
- [ ] Push project to GitHub (`ticklemepark`)

_Patterns and mistakes discovered during development. Max 15 rules. Replace least relevant if full._

1. **bbref abbreviations differ from nba_api in 3 cases**: BRK/BKN (Brooklyn), CHO/CHA (Charlotte), PHO/PHX (Phoenix). Always cross-reference via `BBREF_TO_NBA_API` in `constants.py` when joining datasets across sources. Getting this wrong silently drops rows with no error.

2. **bbref lists visiting team first, home team second** in the scorebox — and the box score table ID encodes the team abbreviation (`box-{ABBR}-game-basic`). Use the scorebox order to assign HOME/AWAY, not the URL alone.

3. **Test scraper parsers directly with fixture HTML**, not by mocking at the module import level. Pass HTML strings straight to `_parse_box_score()` / `_parse_game_urls()`. This is faster, catches HTML structure changes immediately, and keeps fixtures readable in the test file.

4. **Python 3.14 breaks `pip install -e ".[dev]"`** due to a setuptools editable-install backend incompatibility. Install packages individually: `python -m pip install requests beautifulsoup4 pandas ...`. Update pyproject.toml build-system if this needs to be fixed properly.

5. **A full 10-season bbref scrape (~12,300 games) takes ~10 hours** at the 3-second rate limit. Run overnight. The scraper is fault-tolerant — failed games are logged and skipped, not fatal. Check the warning logs after a run to spot systematic failures.

6. **`pandas groupby().apply()` drops the groupby column** in pandas 2.x when the column is part of the DataFrame. Use an explicit `for _, grp in df.groupby("COL"):` loop instead to safely retain the column in the concatenated result. This affects all rolling feature modules (ratings, pace, four_factors).

7. **nba_api imports must be at module level for mocking to work.** Lazy imports inside functions (`from nba_api... import X` inside a def) make `@patch("module.X")` fail with AttributeError since `X` is not a module-level name. Import at the top of the file so the patch target exists.

8. **sklearn 1.8 removed `cv="prefit"` from CalibratedClassifierCV.** Instead, implement calibration manually: get raw probs from the fitted model, fit an `IsotonicRegression` (or 1D `LogisticRegression`) on them, and wrap in a thin `CalibratedModel` class with a `predict_proba()` method. This is more explicit and version-stable.

9. **XGBoost 3.x removed the `use_label_encoder` parameter.** Remove it from any param dict — it silently does nothing but generates a warning that pollutes test output.
