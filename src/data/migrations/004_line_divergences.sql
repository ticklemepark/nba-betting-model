-- Migration 004: Combo line divergence tracking
-- Logs daily instances where Underdog's combo line (PRA, RA) diverges
-- from the sum of its component individual lines.  Used to detect and
-- exploit systematic pricing inefficiencies.

CREATE TABLE IF NOT EXISTS line_divergences (
    id              SERIAL PRIMARY KEY,
    game_date       DATE        NOT NULL,
    player_name     VARCHAR(100) NOT NULL,
    team            VARCHAR(10),
    combo_stat      VARCHAR(10) NOT NULL,   -- 'PRA' or 'RA'
    combo_line      NUMERIC(6,1) NOT NULL,  -- Underdog's direct combo line
    sum_individual  NUMERIC(6,1) NOT NULL,  -- sum of component lines
    divergence      NUMERIC(6,1) NOT NULL,  -- sum_individual - combo_line
    component_lines JSONB,                  -- e.g. {"PTS": 22.5, "REB": 6.5, "AST": 4.5}
    actual_stat     NUMERIC(6,1),           -- filled in by settle script
    closer_to       VARCHAR(20),            -- 'combo', 'sum_individual', 'tie' — filled in after settlement
    logged_at       TIMESTAMP   DEFAULT NOW(),
    UNIQUE (player_name, combo_stat, game_date)
);

CREATE INDEX IF NOT EXISTS idx_line_div_date   ON line_divergences (game_date);
CREATE INDEX IF NOT EXISTS idx_line_div_player ON line_divergences (player_name, combo_stat);
