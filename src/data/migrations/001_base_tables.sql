-- Migration 001: Base tables
--
-- Creates the games table that stores team-level box score data as
-- scraped from Basketball Reference.  Column names match the DataFrame
-- schema produced by src/data/scrapers/bbref.py exactly so that a
-- simple df.to_sql() or bulk INSERT is straightforward.
--
-- Run via: python -c "from src.data.db import execute_sql_file; execute_sql_file('src/data/migrations/001_base_tables.sql')"

-- ---------------------------------------------------------------------------
-- games
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS games (
    id          SERIAL PRIMARY KEY,

    -- Identifiers
    season      INTEGER     NOT NULL,           -- season end year, e.g. 2024
    date        DATE        NOT NULL,
    home        VARCHAR(3)  NOT NULL,           -- Basketball Reference 3-letter abbr
    away        VARCHAR(3)  NOT NULL,

    -- Away team box score totals
    away_mp     VARCHAR(10),                    -- minutes played ("240:00")
    away_fg     SMALLINT,
    away_fga    SMALLINT,
    away_fg_pct NUMERIC(5, 3),
    away_3p     SMALLINT,
    away_3pa    SMALLINT,
    away_3p_pct NUMERIC(5, 3),
    away_ft     SMALLINT,
    away_fta    SMALLINT,
    away_ft_pct NUMERIC(5, 3),
    away_orb    SMALLINT,
    away_drb    SMALLINT,
    away_trb    SMALLINT,
    away_ast    SMALLINT,
    away_stl    SMALLINT,
    away_blk    SMALLINT,
    away_to     SMALLINT,
    away_pf     SMALLINT,
    away_pts    SMALLINT,

    -- Home team box score totals
    home_mp     VARCHAR(10),
    home_fg     SMALLINT,
    home_fga    SMALLINT,
    home_fg_pct NUMERIC(5, 3),
    home_3p     SMALLINT,
    home_3pa    SMALLINT,
    home_3p_pct NUMERIC(5, 3),
    home_ft     SMALLINT,
    home_fta    SMALLINT,
    home_ft_pct NUMERIC(5, 3),
    home_orb    SMALLINT,
    home_drb    SMALLINT,
    home_trb    SMALLINT,
    home_ast    SMALLINT,
    home_stl    SMALLINT,
    home_blk    SMALLINT,
    home_to     SMALLINT,
    home_pf     SMALLINT,
    home_pts    SMALLINT,

    -- Provenance
    bbref_url   TEXT,
    scraped_at  TIMESTAMP   NOT NULL DEFAULT NOW(),

    -- A game is uniquely identified by its date and the two teams.
    -- (Two teams can only meet once per calendar day in the regular season.)
    UNIQUE (date, home, away)
);

CREATE INDEX IF NOT EXISTS idx_games_season ON games (season);
CREATE INDEX IF NOT EXISTS idx_games_date   ON games (date);
CREATE INDEX IF NOT EXISTS idx_games_home   ON games (home);
CREATE INDEX IF NOT EXISTS idx_games_away   ON games (away);
