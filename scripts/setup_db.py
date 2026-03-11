"""PostgreSQL database setup script.

Run this AFTER installing PostgreSQL to:
  1. Create the `nba` database
  2. Apply migration 001 (creates the `games` table)
  3. Optionally seed basic game data from nba_api (fast, no scraping)

Prerequisites:
  1. Install PostgreSQL: https://www.postgresql.org/download/windows/
     - Default port: 5432, default superuser: postgres
  2. Create a .env file in the project root:
       DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/nba
  3. Run this script:
       python scripts/setup_db.py [--seed]

The --seed flag loads game results (home/away/pts/date) for 2015-2024
from nba_api — takes ~30 seconds, no scraping required.  Box score detail
columns (FG, 3P, etc.) will be NULL until you run the full bbref scraper.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv

load_dotenv()

MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "data" / "migrations"


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------

def create_database(admin_url: str, db_name: str = "nba") -> None:
    """Create the target database if it doesn't already exist.

    Connects to the `postgres` admin database to issue CREATE DATABASE.
    admin_url should point to an existing database (e.g. postgresql://postgres:pw@localhost:5432/postgres).
    """
    conn = psycopg2.connect(admin_url)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if cur.fetchone():
        print(f"  Database '{db_name}' already exists — skipping CREATE.")
    else:
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"  Created database '{db_name}'.")
    cur.close()
    conn.close()


def run_migrations(db_url: str) -> None:
    """Apply all migration files in order."""
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        print("  No migration files found.")
        return

    conn = psycopg2.connect(db_url)
    for mf in migration_files:
        print(f"  Applying {mf.name}...", end=" ", flush=True)
        with open(mf) as f:
            sql = f.read()
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("done.")
    conn.close()


# ---------------------------------------------------------------------------
# Optional seed: load game results from nba_api
# ---------------------------------------------------------------------------

def seed_games(db_url: str, seasons: list[int]) -> None:
    """Load basic game results from nba_api into the games table.

    Populates: season, date, home, away, home_pts, away_pts, bbref_url.
    All detailed box score stat columns (fg, 3p, etc.) will be NULL —
    run the bbref scraper to fill those in.
    """
    import pandas as pd
    from nba_api.stats.endpoints import leaguegamelog

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    for season_year in seasons:
        season_str = f"{season_year - 1}-{str(season_year)[2:]}"
        print(f"  Seeding {season_str}...", end=" ", flush=True)

        try:
            log = leaguegamelog.LeagueGameLog(
                season=season_str,
                season_type_all_star="Regular Season",
                timeout=30,
            )
            time.sleep(1)
            df = log.get_data_frames()[0]
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue

        home = df[df["MATCHUP"].str.contains(r"vs\.", regex=True)].copy()
        away = df[df["MATCHUP"].str.contains("@", regex=False)].copy()
        games = (
            home[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS"]]
            .rename(columns={"TEAM_ABBREVIATION": "HOME", "PTS": "HOME_PTS"})
            .merge(
                away[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
                    columns={"TEAM_ABBREVIATION": "AWAY", "PTS": "AWAY_PTS"}
                ),
                on="GAME_ID",
            )
        )
        games["DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date
        games["SEASON"] = season_year

        inserted = 0
        for _, row in games.iterrows():
            try:
                cur.execute(
                    """
                    INSERT INTO games (season, date, home, away, home_pts, away_pts)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, home, away) DO NOTHING
                    """,
                    (
                        int(row["SEASON"]),
                        row["DATE"],
                        row["HOME"],
                        row["AWAY"],
                        int(row["HOME_PTS"]),
                        int(row["AWAY_PTS"]),
                    ),
                )
                inserted += cur.rowcount
            except Exception:
                conn.rollback()
                raise

        conn.commit()
        print(f"{inserted} games inserted.")

    cur.close()
    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the nba database.")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Load game results from nba_api after creating the schema.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(range(2015, 2025)),
        help="Season end years to seed (default: 2015-2024).",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print(
            "ERROR: DATABASE_URL not set.\n"
            "Create a .env file with:\n"
            "  DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/nba"
        )
        sys.exit(1)

    # Build admin URL pointing to 'postgres' database for CREATE DATABASE
    if "/nba" in db_url:
        admin_url = db_url.replace("/nba", "/postgres")
    else:
        admin_url = db_url

    print("Step 1: Creating database...")
    try:
        create_database(admin_url)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print(
            "  Is PostgreSQL running? Start it and try again.\n"
            "  Windows: net start postgresql-x64-17  (adjust version number)"
        )
        sys.exit(1)

    print("Step 2: Running migrations...")
    try:
        run_migrations(db_url)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    if args.seed:
        print(f"Step 3: Seeding game results for seasons {args.seasons}...")
        seed_games(db_url, args.seasons)

    print("\nDone. Next steps:")
    print("  • Run the bbref scraper to fill in full box score stats:")
    print("    python -c \"from src.data.scrapers.bbref import scrape_seasons; import pandas as pd; df = scrape_seasons(list(range(2015,2025))); print(df.shape)\"")
    print("  • Run verification: python scripts/verify_phase1.py")


if __name__ == "__main__":
    main()
