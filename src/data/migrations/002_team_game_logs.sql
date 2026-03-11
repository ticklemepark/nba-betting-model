-- Migration 002: team_game_logs table
--
-- Stores raw per-game box score data for each team fetched via nba_api.
-- One row per team per game.  All abbreviations use nba_api convention
-- (BKN, CHA, PHX), not bbref convention (BRK, CHO, PHO).
--
-- This table is the base for all rolling feature computations (ratings,
-- pace, four_factors, wins_L10, rest_days).  Populate via:
--     from src.data.nba_api_client import fetch_team_game_logs

CREATE TABLE IF NOT EXISTS team_game_logs (
    -- Identifiers
    game_id     VARCHAR(20)  NOT NULL,
    team        VARCHAR(3)   NOT NULL,
    opp         VARCHAR(3)   NOT NULL,
    season      SMALLINT     NOT NULL,
    date        DATE         NOT NULL,
    is_home     BOOLEAN      NOT NULL,

    -- Game outcome
    wl          CHAR(1),                  -- 'W' or 'L'

    -- Game duration (minutes, e.g. 48 regulation, 53 one OT)
    min         NUMERIC(5,2),

    -- Box score totals
    fgm         SMALLINT,
    fga         SMALLINT,
    fg3m        SMALLINT,
    fg3a        SMALLINT,
    ftm         SMALLINT,
    fta         SMALLINT,
    oreb        SMALLINT,
    dreb        SMALLINT,
    reb         SMALLINT,
    ast         SMALLINT,
    tov         SMALLINT,
    stl         SMALLINT,
    blk         SMALLINT,
    pf          SMALLINT,
    pts         SMALLINT,
    plus_minus  SMALLINT,

    -- Provenance
    fetched_at  TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (game_id, team)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_tgl_team_date   ON team_game_logs (team, date);
CREATE INDEX IF NOT EXISTS idx_tgl_season_date ON team_game_logs (season, date);
CREATE INDEX IF NOT EXISTS idx_tgl_game_id     ON team_game_logs (game_id);
