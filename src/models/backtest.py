"""Walk-forward backtesting engine for the game outcome model.

Simulates placing bets on games where our model's predicted edge exceeds a
threshold, tracks P&L, and reports performance metrics.

Since historical Underdog lines are not yet available (those arrive in
Phase 4), the backtest assumes a flat implied probability of 50%
(standard Underdog pick'em). The edge is therefore:

    edge = model_probability - 0.50

P&L is tracked in units (1 unit staked per pick):
    Correct pick:   +1.0 units
    Incorrect pick: -1.0 units

This gives a clean break-even win rate of 50% and ROI equivalent to
win_rate - 0.50.  When real Underdog lines are integrated in Phase 4,
the implied probability and payout terms can be updated directly.

Usage:
    from src.models.backtest import run_game_backtest

    result = run_game_backtest(feature_df, model)
    print(result.summary())
    result.picks_df.to_csv("picks_2024.csv", index=False)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.models.game_outcome import ALL_FEATURE_COLS, GameOutcomeModel, prepare_features
from src.models.calibration import calibration_curve_data

log = logging.getLogger(__name__)

_DEFAULT_EDGE_THRESHOLD = 0.04    # 4% minimum edge to place a bet
_DEFAULT_IMPLIED_PROB   = 0.50    # Underdog pick'em implied probability
_TEST_SEASON            = 2024    # default held-out test season


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """All backtest metrics and per-pick data for one test period."""

    n_games:         int
    n_bets:          int
    win_rate:        float    # bets won / total bets placed
    roi:             float    # net P&L / n_bets  (fraction, not %)
    brier_score:     float    # on all games (model quality regardless of edge)
    log_loss_val:    float    # on all games
    sharpe:          float    # annualised Sharpe ratio of daily P&L
    max_drawdown:    float    # worst peak-to-trough loss in units
    calibration:     pd.DataFrame   # output of calibration_curve_data()
    picks_df:        pd.DataFrame   # per-game details incl. edge, PNL, WIN
    edge_threshold:  float

    def summary(self) -> str:
        pct_games = (
            f"{self.n_bets / self.n_games:.1%}" if self.n_games > 0 else "N/A"
        )
        lines = [
            f"--- Backtest Summary (edge >= {self.edge_threshold:.0%}) ---",
            f"Games evaluated : {self.n_games}",
            f"Bets placed     : {self.n_bets}  ({pct_games} of games)",
            f"Win rate        : {self.win_rate:.1%}" if not np.isnan(self.win_rate) else "Win rate        : N/A (no bets)",
            f"ROI             : {self.roi:+.1%}" if not np.isnan(self.roi) else "ROI             : N/A",
            f"Brier score     : {self.brier_score:.4f}",
            f"Log loss        : {self.log_loss_val:.4f}",
            f"Sharpe ratio    : {self.sharpe:.2f}",
            f"Max drawdown    : {self.max_drawdown:.1f} units",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_game_backtest(
    feature_df: pd.DataFrame,
    model: GameOutcomeModel,
    test_seasons: list[int] | int = _TEST_SEASON,
    edge_threshold: float = _DEFAULT_EDGE_THRESHOLD,
    implied_prob: float = _DEFAULT_IMPLIED_PROB,
) -> BacktestResult:
    """Backtest the game outcome model on held-out test seasons.

    Args:
        feature_df:     Output of build_game_features() with HOME_PTS/AWAY_PTS.
        model:          Trained (optionally calibrated) GameOutcomeModel.
        test_seasons:   Season(s) to evaluate. Default: 2024.
        edge_threshold: Minimum |edge| to place a bet. Default: 4%.
        implied_prob:   Underdog's assumed implied probability. Default: 50%.

    Returns:
        BacktestResult with all metrics and per-pick DataFrame.

    Raises:
        ValueError: If no data exists for the specified test seasons.
    """
    if isinstance(test_seasons, int):
        test_seasons = [test_seasons]

    df = prepare_features(feature_df)
    test = df[df["SEASON"].isin(test_seasons)].copy()

    if test.empty:
        raise ValueError(f"No data found for test seasons {test_seasons}.")

    if "LABEL" not in test.columns:
        raise ValueError(
            "feature_df must contain HOME_PTS and AWAY_PTS to compute LABEL."
        )

    feat_cols = [c for c in ALL_FEATURE_COLS if c in test.columns]
    X_test = test[feat_cols]
    y_test = test["LABEL"]

    proba = model.predict_proba(X_test)[:, 1]   # P(home win)
    edge  = proba - implied_prob                 # positive → bet home; negative → bet away

    # Build per-pick DataFrame
    picks = test[["DATE", "HOME", "AWAY", "SEASON", "LABEL"]].copy()
    picks["HOME_WIN_PROB"] = proba
    picks["EDGE"]          = edge
    picks["BET"]           = (np.abs(edge) >= edge_threshold).astype(int)
    picks["BET_HOME"]      = (edge >= 0).astype(int)   # 1 = bet home, 0 = bet away
    picks["WIN"] = np.where(
        picks["BET"] == 0,
        np.nan,
        (picks["BET_HOME"] == picks["LABEL"]).astype(float),
    )
    picks["PNL"] = picks.apply(
        lambda r: (1.0 if r["WIN"] == 1.0 else -1.0) if r["BET"] == 1 else 0.0,
        axis=1,
    )
    picks = picks.sort_values("DATE").reset_index(drop=True)

    # Aggregate metrics
    bets     = picks[picks["BET"] == 1]
    n_bets   = len(bets)
    win_rate = float(bets["WIN"].mean())  if n_bets > 0 else float("nan")
    roi      = float(bets["PNL"].sum() / n_bets) if n_bets > 0 else float("nan")

    brier = float(brier_score_loss(y_test, proba))
    ll    = float(log_loss(y_test, proba))

    daily_pnl = picks.groupby("DATE")["PNL"].sum()
    sharpe    = _sharpe_ratio(daily_pnl)
    max_dd    = _max_drawdown(picks["PNL"])

    cal = calibration_curve_data(y_test, proba)

    log.info(
        "Backtest on %d games: %d bets, win_rate=%.3f, ROI=%+.3f",
        len(test), n_bets, win_rate if not np.isnan(win_rate) else 0, roi if not np.isnan(roi) else 0,
    )

    return BacktestResult(
        n_games=len(test),
        n_bets=n_bets,
        win_rate=win_rate,
        roi=roi,
        brier_score=brier,
        log_loss_val=ll,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calibration=cal,
        picks_df=picks,
        edge_threshold=edge_threshold,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sharpe_ratio(daily_pnl: pd.Series, periods_per_year: int = 82) -> float:
    """Annualised Sharpe ratio of daily P&L (82 NBA regular-season days)."""
    if daily_pnl.empty or daily_pnl.std() == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(periods_per_year))


def _max_drawdown(pnl_series: pd.Series) -> float:
    """Maximum peak-to-trough cumulative loss in units."""
    if pnl_series.empty:
        return 0.0
    cumulative = pnl_series.cumsum()
    peak       = cumulative.cummax()
    drawdown   = peak - cumulative
    return float(drawdown.max())
