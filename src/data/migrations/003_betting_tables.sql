-- Migration 003: Betting tables for Underdog Fantasy integration
-- Run with: python -c "from src.data.db import execute_sql_file; execute_sql_file('src/data/migrations/003_betting_tables.sql')"

-- Stores fetched Underdog lines (prop and game) for each day.
CREATE TABLE IF NOT EXISTS underdog_lines (
    id           SERIAL PRIMARY KEY,
    stat_type    VARCHAR(20)  NOT NULL,     -- 'OVER_UNDER' or 'RIVAL'
    player_id    VARCHAR(40),               -- NULL for game (rival) lines
    player_name  VARCHAR(100),
    team         VARCHAR(10),
    opp          VARCHAR(10),
    game_id      VARCHAR(40)  NOT NULL,
    stat         VARCHAR(20)  NOT NULL,     -- 'PTS', 'REB', 'PRA', 'GAME', etc.
    line         NUMERIC(6,2),             -- NULL for game lines
    over_payout  NUMERIC(5,3),             -- Underdog implied prob for OVER / home
    under_payout NUMERIC(5,3),             -- Underdog implied prob for UNDER / away
    game_date    DATE         NOT NULL,
    fetched_at   TIMESTAMP    DEFAULT NOW()
);

-- Unique constraint: one row per player/stat/date (game lines: player_id IS NULL)
CREATE UNIQUE INDEX IF NOT EXISTS uq_underdog_lines
    ON underdog_lines (game_id, COALESCE(player_id, ''), stat, game_date);

CREATE INDEX IF NOT EXISTS ix_underdog_lines_date  ON underdog_lines (game_date);
CREATE INDEX IF NOT EXISTS ix_underdog_lines_team  ON underdog_lines (team, game_date);


-- Stores each bet entry placed (2-6 picks).
CREATE TABLE IF NOT EXISTS bet_entries (
    id                 SERIAL        PRIMARY KEY,
    entry_ref          VARCHAR(40)   UNIQUE NOT NULL,   -- UUID generated at log time
    entry_size         SMALLINT      NOT NULL CHECK (entry_size BETWEEN 2 AND 6),
    payout_multiplier  NUMERIC(5,2)  NOT NULL,          -- 3, 6, 10, 20, or 36
    bet_amount         NUMERIC(10,2) NOT NULL CHECK (bet_amount > 0),
    placed_at          TIMESTAMP     NOT NULL,
    game_date          DATE          NOT NULL,
    status             VARCHAR(20)   NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending','won','lost')),
    result_amount      NUMERIC(10,2),                   -- NULL until settled
    settled_at         TIMESTAMP,
    notes              TEXT
);

CREATE INDEX IF NOT EXISTS ix_bet_entries_date   ON bet_entries (game_date);
CREATE INDEX IF NOT EXISTS ix_bet_entries_status ON bet_entries (status);


-- Stores each individual pick within a bet entry.
CREATE TABLE IF NOT EXISTS entry_picks (
    id               SERIAL        PRIMARY KEY,
    entry_ref        VARCHAR(40)   NOT NULL REFERENCES bet_entries(entry_ref) ON DELETE CASCADE,
    game_id          VARCHAR(40),
    player_name      VARCHAR(100),
    team             VARCHAR(10),
    stat             VARCHAR(20)   NOT NULL,   -- 'PTS', 'REB', 'GAME', etc.
    direction        VARCHAR(10)   NOT NULL CHECK (direction IN ('over','under','home','away')),
    line             NUMERIC(6,2),
    model_prediction NUMERIC(6,2),             -- model median (props) or NULL (games)
    edge             NUMERIC(6,3)  NOT NULL,   -- model_prob - implied_prob
    confidence       NUMERIC(5,3)              -- P(direction correct) from quantile model
);

CREATE INDEX IF NOT EXISTS ix_entry_picks_entry ON entry_picks (entry_ref);
