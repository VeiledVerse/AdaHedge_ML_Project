"""
================================================================================
  Adaptive Hedge
  Machine Learning (Nicolò Cesa-Bianchi)
  University of Milan, A.Y. 2025/26

  Extends the AdaHedge engine into a quantitative portfolio strategy using 
  real-world US Sector ETF data.

  Design philosophy:
    • Experts are Traders/Position Allocators, not Forecasters.
    • Each expert e = (strategy × asset) outputs a continuous position
      signal s_t^e in [-1, 1] representing long/short exposure.
    • Daily trading return: tradingReturn_t^e = s_t^e · r_t^{asset(e)}.
    • The affine loss mapping l = (R_max - ret)/(2R_max) ensures that
      minimising cumulative loss == maximising cumulative trading return.
    • The Hedge/AdaHedge functions are imported WITHOUT modification.

  Key design notes:
    • tanh signal transform with annual expanding-window recalibration
      (avoids look-ahead; prevents COVID/rate-hike saturation).
    • R_max uses the 99.5th percentile of |return| to prevent one March-2020
      tail day from compressing all other days' losses toward 0.5.
    • Transaction costs are post-hoc on NET position changes only
      (cost-aware online learning is a different theoretical problem;
       see the design note in SECTION 6).
    • Sharpe ratios computed with risk-free rate = 0. The zero-rate
      assumption inflates absolute Sharpe numbers during 2022–23
      (T-Bill ~4-5%), but the shift is EQUAL across all strategies,
      so relative comparisons remain valid.

  Reference: 
    • "Adaptive Hedge" — van Erven et al., NeurIPS 2011.
    • "Prediction, Learning, and Games" — Nicolò Cesa-Bianchi & Gábor Lugosi, 2006.  
================================================================================
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
# ── Import Phase 1 algorithm runners ─────────────────────────────────────────
from adahedge import (
    run_hedge_fixed_learning_rate,
    run_adahedge_algorithm,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — CONFIGURATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PHASE2_RANDOM_SEED = 777
numpy_rng_phase2   = np.random.default_rng(PHASE2_RANDOM_SEED)

# ── Directory paths ────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(_SCRIPT_DIR)
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "Plots")
DATASETS_DIR  = os.path.join(PROJECT_ROOT, "Datasets")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "Reports")
for _directory in (PLOTS_DIR, DATASETS_DIR, REPORTS_DIR):
    os.makedirs(_directory, exist_ok=True)

# ── Asset universe: 9 US Sector SPDR ETFs ─────────────────────────────────────
SECTOR_ETF_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB"]
SECTOR_ETF_DISPLAY_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Cons. Discretionary",
    "XLI": "Industrials",
    "XLP": "Cons. Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
}
NUMBER_OF_ASSETS_IN_BASKET = len(SECTOR_ETF_TICKERS)   # 9

# ── Trading strategy types ─────────────────────────────────────────────────────
STRATEGY_MOMENTUM   = "Momentum_20d"
STRATEGY_MEANREV    = "MeanRev_20d"
STRATEGY_CROSSOVER  = "Trend_SMA10_50"
STRATEGY_BREAKOUT   = "Breakout_20d"
ALL_STRATEGY_TYPES  = [STRATEGY_MOMENTUM, STRATEGY_MEANREV, STRATEGY_CROSSOVER, STRATEGY_BREAKOUT]
NUMBER_OF_STRATEGY_TYPES = len(ALL_STRATEGY_TYPES)    # 4

# Total experts = strategies × assets = 4 × 9 = 36
TOTAL_NUMBER_OF_EXPERTS = NUMBER_OF_STRATEGY_TYPES * NUMBER_OF_ASSETS_IN_BASKET

# Build expert index lookup tables (used everywhere for broadcasting)
expert_label_list             = []
expert_asset_index_list       = []
expert_strategy_index_list    = []
for _strategy_idx, _strategy_name in enumerate(ALL_STRATEGY_TYPES):
    for _asset_idx, _ticker in enumerate(SECTOR_ETF_TICKERS):
        expert_label_list.append(f"{_strategy_name}|{_ticker}")
        expert_asset_index_list.append(_asset_idx)
        expert_strategy_index_list.append(_strategy_idx)

EXPERT_LABEL_ARRAY            = np.array(expert_label_list)
EXPERT_ASSET_INDEX_ARRAY      = np.array(expert_asset_index_list,    dtype=int)
EXPERT_STRATEGY_INDEX_ARRAY   = np.array(expert_strategy_index_list, dtype=int)

# ── Data date range ────────────────────────────────────────────────────────────
DATA_DOWNLOAD_START_DATE = "2019-01-01"
DATA_DOWNLOAD_END_DATE   = "2026-06-30"

# ── Signal calibration ─────────────────────────────────────────────────────────
# We use the first 252 trading days (1 year) as a warm-up period to calculate
# typical indicator values. We recompute these scales every 252 days using an
# expanding window of past data. This prevents us from using future data and
# helps the strategy adapt when markets become highly volatile (like in 2020 or 2022).
BURN_IN_NUMBER_OF_TRADING_DAYS      = 252
CALIBRATION_UPDATE_INTERVAL_IN_DAYS = 252

# ── Transaction cost configuration ────────────────────────────────────────────
# We set a base fee of 10 basis points (0.1%) for every trade. To see how
# transaction costs affect our strategy's performance, we run backtests across
# different cost levels: 0% (free), 0.05% (5 bp), 0.1% (10 bp), and 0.25% (25 bp).
TRANSACTION_COST_BASE_COEFFICIENT = 0.001
TRANSACTION_COST_SWEEP_LEVELS     = [0.0, 0.0005, 0.001, 0.0025]   # {0, 5, 10, 25 bp}

# ── Loss mapping R_max ─────────────────────────────────────────────────────────
# We clip daily returns at the 99.5th percentile instead of using the absolute maximum.
# This prevents a single extreme market day (like the March 2020 crash) from shrinking
# all other daily losses close to 0.5, which would make it hard for AdaHedge to tell
# the difference between good and bad trades.
RETURN_BOUND_CLIPPING_PERCENTILE = 99.5

# ── Portfolio initialisation ───────────────────────────────────────────────────
# We start the backtest with an initial capital of $1000.0.
INITIAL_PORTFOLIO_WEALTH_DOLLARS = 1000.0

# ── Rolling Sharpe window ──────────────────────────────────────────────────────
# We calculate the rolling Sharpe ratio over a window of 60 trading days (about 3 months)
# to see how risk-adjusted performance changes over time.
ROLLING_SHARPE_WINDOW_SIZE_IN_DAYS = 60

# ── Risk-free rate for Sharpe ──────────────────────────────────────────────────
# We assume the risk-free return rate is 0. This is done to simplify calculations.
# While it slightly inflates the Sharpe ratios when interest rates are high, the
# effect is the same for all strategies, so the relative comparisons remain fair.
ANNUAL_RISK_FREE_RATE = 0.0
DAILY_RISK_FREE_RATE  = ANNUAL_RISK_FREE_RATE / 252.0

TRADING_DAYS_PER_YEAR = 252


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DARK-THEME STYLE
# ══════════════════════════════════════════════════════════════════════════════

_BG    = "#0A0A14"
_PANEL = "#0F0F1F"
_SPINE = "#252540"
_GRID  = "#1A1A30"
_LABEL = "#D0D0EE"
_TITLE = "#EEEEFF"

plt.rcParams.update({
    "figure.facecolor":  _BG,    "axes.facecolor":    _PANEL,
    "axes.edgecolor":    _SPINE, "axes.labelcolor":   _LABEL,
    "axes.titlecolor":   _TITLE, "text.color":        _LABEL,
    "xtick.color":       _LABEL, "ytick.color":       _LABEL,
    "xtick.direction":   "out",  "ytick.direction":   "out",
    "grid.color":        _GRID,  "grid.linestyle":    "-",
    "grid.alpha":        1.0,    "grid.linewidth":    0.6,
    "legend.facecolor":  _BG,    "legend.edgecolor":  _SPINE,
    "legend.labelcolor": _LABEL, "legend.framealpha": 0.90,
    "font.family":       "sans-serif", "font.size":   10,
    "axes.titlesize":    12,     "axes.labelsize":    10,
    "axes.titlepad":     10,     "lines.linewidth":   1.8,
    "lines.antialiased": True,   "figure.dpi":        150,
    "savefig.dpi":       150,    "savefig.facecolor": _BG,
})

# ── Benchmark colour palette ───────────────────────────────────────────────────
COLOR_ADAHEDGE        = "#F72585"    # Hot magenta   — hero algorithm
COLOR_BUYHOLD_EW      = "#06D6A0"    # Emerald teal  — passive baseline
COLOR_BEST_EXPERT     = "#FFD166"    # Golden yellow — hindsight oracle
COLOR_UNIFORM_CRP     = "#4CC9F0"    # Icy cyan      — equal-weight reference
COLOR_HEDGE_CONSERV   = "#B5E48C"    # Sage green    — eta=0.05 (very conservative)
COLOR_HEDGE_MODERATE  = "#FF9F1C"    # Warm amber    — eta=0.5
COLOR_HEDGE_WORST     = "#4361EE"    # Indigo        — worst-case optimal eta
COLOR_THEORETICAL     = "#666680"    # Muted grey    — theory bound reference

BENCHMARK_COLOR_MAP = {
    "AdaHedge (phi=2)":        COLOR_ADAHEDGE,
    "Buy-and-Hold EW":         COLOR_BUYHOLD_EW,
    "Best Expert (Hindsight)": COLOR_BEST_EXPERT,
    "Uniform CRP":             COLOR_UNIFORM_CRP,
    "Hedge eta=0.05":          COLOR_HEDGE_CONSERV,
    "Hedge eta=0.5":           COLOR_HEDGE_MODERATE,
    "Hedge eta=eta*(T)":       COLOR_HEDGE_WORST,
}


def _apply_dark_axis_styling(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent dark-theme styling to a matplotlib axis."""
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_linewidth(0.8)
    ax.grid(True, which="major", linewidth=0.6)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, pad=4)


def _save_plot_to_directory(figure_handle, filename_without_extension: str) -> None:
    """Save figure to Plots/ and close the handle."""
    full_output_path = os.path.join(PLOTS_DIR, filename_without_extension + ".png")
    figure_handle.savefig(full_output_path, bbox_inches="tight")
    plt.close(figure_handle)
    print(f"  [SAVED] {full_output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def download_or_load_etf_adjusted_close_prices() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Download daily adjusted-close prices for all 9 sector ETFs via yfinance.
    Caches the result to Datasets/etf_daily_prices.csv to avoid re-downloading
    on subsequent runs.  If the cache file is present and contains all 9 tickers,
    it is loaded directly.

    Args:
        None (uses global constants DATA_DOWNLOAD_START_DATE, DATA_DOWNLOAD_END_DATE,
              SECTOR_ETF_TICKERS).

    Returns:
        (price_matrix, full_date_index):
            price_matrix    : shape (T+1, N_assets) float64 — adjusted close prices.
                              Row 0 = first available price; row T = final price.
            full_date_index : DatetimeIndex of length T+1 — one per price row.
    """
    cache_file_path = os.path.join(DATASETS_DIR, "etf_daily_prices.csv")

    if os.path.exists(cache_file_path):
        print(f"\n  [DATA] Loading cached prices -> {cache_file_path}")
        cached_price_dataframe = pd.read_csv(
            cache_file_path, index_col=0, parse_dates=True
        )
        all_tickers_present = all(
            ticker in cached_price_dataframe.columns for ticker in SECTOR_ETF_TICKERS
        )
        if all_tickers_present:
            ordered_price_dataframe = cached_price_dataframe[SECTOR_ETF_TICKERS]
            print(
                f"  [DATA] {len(ordered_price_dataframe)} rows | "
                f"{ordered_price_dataframe.index[0].date()} -> "
                f"{ordered_price_dataframe.index[-1].date()}"
            )
            return np.array(ordered_price_dataframe.values), pd.DatetimeIndex(ordered_price_dataframe.index)
        print("  [DATA] Cache is incomplete — re-downloading ...")

    print(
        f"\n  [DATA] Downloading from yfinance: {', '.join(SECTOR_ETF_TICKERS)}"
        f"\n         {DATA_DOWNLOAD_START_DATE} -> {DATA_DOWNLOAD_END_DATE}"
    )

    raw_downloaded_data = yf.download(
        tickers=SECTOR_ETF_TICKERS,
        start=DATA_DOWNLOAD_START_DATE,
        end=DATA_DOWNLOAD_END_DATE,
        auto_adjust=True,               #Returns prices already adjusted for splits and dividends
        progress=False,
        threads=False,
    )

    # Check if download was successful
    if raw_downloaded_data is None or raw_downloaded_data.empty:
        raise ValueError(
            f"Failed to download data from yfinance for tickers: {SECTOR_ETF_TICKERS}"
        )

    # Extract adjusted close prices, handling both flat and MultiIndex columns.
    if isinstance(raw_downloaded_data.columns, pd.MultiIndex):
        level_zero_labels = raw_downloaded_data.columns.get_level_values(0).unique().tolist()
        if "Close" in level_zero_labels:
            price_dataframe = raw_downloaded_data["Close"][SECTOR_ETF_TICKERS]
        elif "Adj Close" in level_zero_labels:
            price_dataframe = raw_downloaded_data["Adj Close"][SECTOR_ETF_TICKERS]
        else:
            raise ValueError(
                f"Cannot locate Close prices in downloaded data. "
                f"Top-level column labels: {level_zero_labels}"
            )
    else:
        # Single-ticker download (should not happen here, but handled for safety)
        price_dataframe = raw_downloaded_data[SECTOR_ETF_TICKERS]

    # Forward-fill small gaps (e.g., ETFs trading on different holiday schedules)
    price_dataframe = price_dataframe.ffill()

    # Drop any remaining NaN rows (typically the very first row if data starts
    # on different dates for different ETFs)
    rows_before_drop = len(price_dataframe)
    price_dataframe  = price_dataframe.dropna()
    rows_dropped     = rows_before_drop - len(price_dataframe)
    if rows_dropped > 0:
        print(f"  [DATA] Dropped {rows_dropped} leading NaN row(s) after forward-fill")

    # Cache to disk for subsequent runs
    price_dataframe.to_csv(cache_file_path)
    print(
        f"  [DATA] Cached -> {cache_file_path}\n"
        f"  [DATA] {len(price_dataframe)} trading days | "
        f"{price_dataframe.index[0].date()} -> {price_dataframe.index[-1].date()}"
    )

    return np.array(price_dataframe.values), pd.DatetimeIndex(price_dataframe.index)


def compute_daily_simple_returns_from_prices(
    price_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute daily simple returns from a price matrix.

    r_t^a = (P_t^a - P_{t-1}^a) / P_{t-1}^a

    Args:
        price_matrix : shape (T+1, N_assets).

    Returns:
        daily_return_matrix : shape (T, N_assets).
    """
    return (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SIGNAL COMPUTATION (LOOK-AHEAD SAFE)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_tanh_scale_parameter_from_indicator_history(
    indicator_values_seen_so_far: np.ndarray,
    fallback_scale: float = 1.0,
) -> float:
    """
    Compute the tanh scaling parameter: scale = 1 / std(indicator_values).

    Rationale: setting scale = 1/std means that a 1-standard-deviation indicator
    move maps to tanh(1) ~= 0.76 — using most of the [-1, 1] output range
    without saturating on typical moves, while still saturating at extremes.

    Falls back to fallback_scale if std is near zero (degenerate indicator).
    """
    indicator_standard_deviation = float(np.std(indicator_values_seen_so_far))
    if indicator_standard_deviation < 1e-10:
        return fallback_scale
    return 1.0 / indicator_standard_deviation


def compute_signals_for_single_asset_with_expanding_window_calibration(
    adjusted_close_prices: np.ndarray,
    total_number_of_return_days: int,
) -> dict:
    """
    Compute continuous trading signals in [-1, 1] for all 4 strategy types
    on a single asset, using expanding-window tanh scale calibration.

    LOOK-AHEAD GUARANTEE
    Price indexing convention:
        prices[t] = adjusted close price at the END of calendar day t.
        return[t] = (prices[t+1] - prices[t]) / prices[t]  for t = 0 … T-1.

    The SIGNAL for return day t (0-indexed) uses prices[0 … t] only
    (price_index = t), i.e., it sees the close BEFORE today's return
    is realised.  No signal ever reads prices[t+1] or beyond.

    EXPANDING-WINDOW CALIBRATION
    The tanh scale parameter (= 1/std(indicator)) is computed at day
    BURN_IN_NUMBER_OF_TRADING_DAYS using all indicator values 0 … t-1,
    then re-computed every CALIBRATION_UPDATE_INTERVAL_IN_DAYS days
    (expanding window — never looks forward).

    This prevents the 2019-only burn-in from freezing scales at a
    low-volatility baseline: the 2020 COVID crash and 2022 rate-hike
    shock both sharply expand indicator std, and the annual recalibration
    adapts the scale downward so signals do NOT saturate near ±1 for
    months at a time.

    Args:
        adjusted_close_prices           : shape (T+1,) — prices[0] through prices[T].
        total_number_of_return_days     : T.

    Returns:
        dict: strategy_name -> signal_array of shape (T,).
              Values are NaN where indicator history is insufficient.
    """
    total_days_of_returns = total_number_of_return_days
    prices                = adjusted_close_prices         # alias for clarity

    # Output arrays: NaN = "no signal yet" (will be replaced by 0 in backtest)
    signals_by_strategy = {strategy: np.full(total_days_of_returns, np.nan)
                           for strategy in ALL_STRATEGY_TYPES}

    # Raw (unscaled, pre-tanh) indicator values — needed for calibration
    raw_momentum_indicator_values        = np.full(total_days_of_returns, np.nan)
    raw_mean_reversion_indicator_values  = np.full(total_days_of_returns, np.nan)
    raw_crossover_indicator_values       = np.full(total_days_of_returns, np.nan)

    # Strategy lookback constants
    MOMENTUM_LOOKBACK_IN_DAYS     = 20   # 20-day price return indicator
    MEAN_REVERSION_LOOKBACK_DAYS  = 20   # 20-day rolling mean/std for Z-score
    SHORT_SMA_PERIOD_IN_DAYS      = 10   # fast moving average
    LONG_SMA_PERIOD_IN_DAYS       = 50   # slow moving average
    BREAKOUT_CHANNEL_LOOKBACK_DAYS= 20   # Donchian channel width

    # ── Pre-compute raw indicator values (no calibration yet) ──────────────────
    for return_day_index in range(total_days_of_returns):
        # price_index = return_day_index: the closing price BEFORE today's return
        price_index = return_day_index

        # ── Strategy 1: Momentum ──────────────────────────────────────────────
        # Raw indicator: 20-day log price return.
        # Signal direction: positive momentum -> long (tanh positive)
        if price_index >= MOMENTUM_LOOKBACK_IN_DAYS:
            lookback_price  = prices[price_index - MOMENTUM_LOOKBACK_IN_DAYS]
            current_price   = prices[price_index]
            raw_momentum_indicator_values[return_day_index] = (
                (current_price - lookback_price) / lookback_price
            )

        # ── Strategy 2: Mean Reversion (Z-score) ─────────────────────────────
        # Raw indicator: (P_t - μ_20) / σ_20
        # Signal direction: NEGATIVE sign — buy when price is below mean.
        if price_index >= MEAN_REVERSION_LOOKBACK_DAYS - 1:
            rolling_price_window = prices[
                price_index - MEAN_REVERSION_LOOKBACK_DAYS + 1 : price_index + 1
            ]
            window_mean = float(np.mean(rolling_price_window))
            window_std  = float(np.std(rolling_price_window))
            if window_std > 1e-10:
                raw_mean_reversion_indicator_values[return_day_index] = (
                    (prices[price_index] - window_mean) / window_std
                )

        # ── Strategy 3: Trend Crossover (SMA10 vs SMA50) ─────────────────────
        # Raw indicator: (SMA10 - SMA50) / SMA50  (relative spread)
        # Signal direction: positive spread -> long (short MA above long MA)
        if price_index >= LONG_SMA_PERIOD_IN_DAYS - 1:
            sma_short = float(np.mean(
                prices[price_index - SHORT_SMA_PERIOD_IN_DAYS + 1 : price_index + 1]
            ))
            sma_long  = float(np.mean(
                prices[price_index - LONG_SMA_PERIOD_IN_DAYS + 1 : price_index + 1]
            ))
            if sma_long > 1e-10:
                raw_crossover_indicator_values[return_day_index] = (
                    (sma_short - sma_long) / sma_long
                )

        # ── Strategy 4: Breakout (Donchian Channel) ───────────────────────────
        # Signal in [-1, +1] by construction — no tanh needed.
        # Formula: 2 * (P - L_20) / (H_20 - L_20) - 1
        # Interpretation: +1 = at channel high (breakout long), -1 = at channel low
        if price_index >= BREAKOUT_CHANNEL_LOOKBACK_DAYS - 1:
            channel_prices = prices[
                price_index - BREAKOUT_CHANNEL_LOOKBACK_DAYS + 1 : price_index + 1
            ]
            channel_high  = float(np.max(channel_prices))
            channel_low   = float(np.min(channel_prices))
            channel_range = channel_high - channel_low
            if channel_range > 1e-10:
                signals_by_strategy[STRATEGY_BREAKOUT][return_day_index] = (
                    2.0 * (prices[price_index] - channel_low) / channel_range - 1.0
                )
            else:
                # Flat channel (price constant) -> no directional signal
                signals_by_strategy[STRATEGY_BREAKOUT][return_day_index] = 0.0

    # ── Apply expanding-window tanh calibration ────────────────────────────────
    # We update the scaling factor of our indicators at regular checkpoints:
    #   1. First calibration: after the 1-year warm-up period (day 252).
    #   2. Annual updates: every 252 days after that.
    # At each checkpoint, we calculate the standard deviation using all data collected
    # since the beginning (expanding window). This ensures we never look ahead.
    # Before the first checkpoint, we use a default scale of 1.0.
    current_momentum_tanh_scale   = 1.0
    current_mean_rev_tanh_scale   = 1.0
    current_crossover_tanh_scale  = 1.0

    # The next checkpoint where scales will be updated
    next_calibration_checkpoint_day = BURN_IN_NUMBER_OF_TRADING_DAYS

    for return_day_index in range(total_days_of_returns):

        # Check if we have reached a calibration checkpoint
        if return_day_index == next_calibration_checkpoint_day:
            # Collect all valid (non-NaN) indicator values BEFORE today
            valid_momentum_history = raw_momentum_indicator_values[:return_day_index]
            valid_momentum_history = valid_momentum_history[~np.isnan(valid_momentum_history)]

            valid_mean_rev_history = raw_mean_reversion_indicator_values[:return_day_index]
            valid_mean_rev_history = valid_mean_rev_history[~np.isnan(valid_mean_rev_history)]

            valid_crossover_history = raw_crossover_indicator_values[:return_day_index]
            valid_crossover_history = valid_crossover_history[~np.isnan(valid_crossover_history)]

            # Update scales (require at least 10 valid observations)
            if len(valid_momentum_history) >= 10:
                current_momentum_tanh_scale = (
                    _compute_tanh_scale_parameter_from_indicator_history(valid_momentum_history)
                )
            if len(valid_mean_rev_history) >= 10:
                current_mean_rev_tanh_scale = (
                    _compute_tanh_scale_parameter_from_indicator_history(valid_mean_rev_history)
                )
            if len(valid_crossover_history) >= 10:
                current_crossover_tanh_scale = (
                    _compute_tanh_scale_parameter_from_indicator_history(valid_crossover_history)
                )

            next_calibration_checkpoint_day += CALIBRATION_UPDATE_INTERVAL_IN_DAYS

        # Apply calibrated tanh transforms for the three tanh-based strategies
        if not np.isnan(raw_momentum_indicator_values[return_day_index]):
            signals_by_strategy[STRATEGY_MOMENTUM][return_day_index] = np.tanh(
                current_momentum_tanh_scale * raw_momentum_indicator_values[return_day_index]
            )

        if not np.isnan(raw_mean_reversion_indicator_values[return_day_index]):
            # Negative sign: mean reversion -> buy when price is BELOW the mean
            signals_by_strategy[STRATEGY_MEANREV][return_day_index] = np.tanh(
                -current_mean_rev_tanh_scale * raw_mean_reversion_indicator_values[return_day_index]
            )

        if not np.isnan(raw_crossover_indicator_values[return_day_index]):
            signals_by_strategy[STRATEGY_CROSSOVER][return_day_index] = np.tanh(
                current_crossover_tanh_scale * raw_crossover_indicator_values[return_day_index]
            )

    return signals_by_strategy


def build_full_expert_signal_matrix(price_matrix: np.ndarray) -> np.ndarray:
    """
    Build the complete (T, K) expert signal matrix for all 36 experts.

    For each asset, calls compute_signals_for_single_asset_... and stacks
    the results by expert index.  NaN signals (warm-up period) are replaced
    with 0.0 — the neutral "no position" action.

    Args:
        price_matrix : shape (T+1, N_assets) — adjusted close prices.

    Returns:
        expert_signal_matrix : shape (T, K) — signals in [-1, 1], NaN replaced by 0.
    """
    total_price_rows = price_matrix.shape[0]
    total_number_of_return_days = total_price_rows - 1

    expert_signal_matrix = np.zeros(
        (total_number_of_return_days, TOTAL_NUMBER_OF_EXPERTS), dtype=np.float64
    )

    print("\n  [SIGNALS] Computing signals for all 36 experts ...")
    for asset_index, ticker_symbol in enumerate(SECTOR_ETF_TICKERS):
        asset_price_series = price_matrix[:, asset_index]
        signals_for_this_asset = (
            compute_signals_for_single_asset_with_expanding_window_calibration(
                adjusted_close_prices=asset_price_series,
                total_number_of_return_days=total_number_of_return_days,
            )
        )
        for strategy_index, strategy_name in enumerate(ALL_STRATEGY_TYPES):
            expert_index = strategy_index * NUMBER_OF_ASSETS_IN_BASKET + asset_index
            raw_signal_array = signals_for_this_asset[strategy_name]
            # Replace NaN with 0 (no position during warm-up)
            expert_signal_matrix[:, expert_index] = np.where(
                np.isnan(raw_signal_array), 0.0, raw_signal_array
            )

    number_of_zero_signal_days = int(np.sum(np.all(expert_signal_matrix == 0, axis=1)))
    print(
        f"  [SIGNALS] Matrix shape : {expert_signal_matrix.shape}\n"
        f"  [SIGNALS] Signal range : [{expert_signal_matrix.min():.4f}, "
        f"{expert_signal_matrix.max():.4f}]\n"
        f"  [SIGNALS] Zero-signal days (warm-up): {number_of_zero_signal_days}"
    )
    return expert_signal_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RETURNS-TO-LOSS MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def compute_expert_trading_return_matrix(
    expert_signal_matrix: np.ndarray,
    asset_daily_return_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the daily trading return for each expert.

    tradingReturn_t^e = signal_t^e × return_t^{asset(e)}

    The expert holds position signal_t^e (set BEFORE day t's return is
    revealed) and earns that position times the realised asset return.

    Args:
        expert_signal_matrix      : (T, K) signals in [-1, 1].
        asset_daily_return_matrix : (T, N_assets) daily simple returns.

    Returns:
        expert_trading_return_matrix : (T, K) — daily P&L fraction per expert.
    """
    expert_trading_return_matrix = np.zeros_like(expert_signal_matrix)
    for expert_index in range(TOTAL_NUMBER_OF_EXPERTS):
        asset_index = EXPERT_ASSET_INDEX_ARRAY[expert_index]
        expert_trading_return_matrix[:, expert_index] = (
            expert_signal_matrix[:, expert_index]
            * asset_daily_return_matrix[:, asset_index]
        )
    return expert_trading_return_matrix


def compute_loss_mapping_return_bound(
    expert_trading_return_matrix: np.ndarray,
    clipping_percentile: float = RETURN_BOUND_CLIPPING_PERCENTILE,
) -> tuple:
    """
    Compute R_max for the affine loss mapping, with percentile clipping.

    See the design note in SECTION 0 for the rationale behind clipping.
    Both the global maximum and the percentile-clipped value are printed
    so the caller can inspect whether a single tail event is dominating.

    Returns:
        (global_r_max, clipped_r_max) — both positive floats.
    """
    all_absolute_trading_returns = np.abs(expert_trading_return_matrix).ravel()
    global_r_max   = float(np.max(all_absolute_trading_returns))
    clipped_r_max  = float(np.percentile(all_absolute_trading_returns, clipping_percentile))
    saturation_rate = float(np.mean(all_absolute_trading_returns > clipped_r_max))

    print(
        f"\n  [R_MAX]  Global max |return|    = {global_r_max*100:.4f}%\n"
        f"  [R_MAX]  {clipping_percentile}th pctile clip    = {clipped_r_max*100:.4f}%\n"
        f"  [R_MAX]  Tail-compression ratio    = {global_r_max / clipped_r_max:.2f}x  "
        f"(global / clipped)\n"
        f"  [R_MAX]  Saturating pairs    = {saturation_rate*100:.3f}% of "
        f"(day × expert) pairs\n"
        f"  [R_MAX]  -> Using {clipping_percentile}th-pctile R_max    = {clipped_r_max*100:.4f}%"
    )
    return global_r_max, clipped_r_max


def map_trading_returns_to_unit_interval_loss_matrix(
    expert_trading_return_matrix: np.ndarray,
    maximum_return_bound: float,
) -> np.ndarray:
    """
    Map trading returns in [-R_max, R_max] to losses in [0, 1].

    Affine rescaling:
        l_t^e = (R_max - tradingReturn_t^e) / (2 · R_max)

    Key properties:
        tradingReturn = +R_max  ->  l = 0   (best possible: zero loss)
        tradingReturn = -R_max  ->  l = 1   (worst possible: maximum loss)
        tradingReturn =  0      ->  l = 0.5 (neutral position)

    PROOF (loss-minimisation == return-maximisation):
        Sum_t l_t^e = T/2 - (1/(2·R_max)) · Sum_t tradingReturn_t^e

        Since T/2 is a constant shared by ALL experts, and 1/(2·R_max) > 0:
            argmin_e Sum_t l_t^e  ==  argmax_e Sum_t tradingReturn_t^e   ∎

        This equivalence is EXACT regardless of:
          • Varying daily volatility (R_max is a fixed global constant).
          • Asset-specific baseline volatilities (same R_max for all assets).
        The choice of global vs asset-specific R_max shifts the LOSS RANGE
        assigned to each asset's returns, but does not change the argmax
        or the loss ordering among experts trading the same asset.

    Note on tail clipping: when using the percentile-clipped R_max,
        returns beyond ±R_max are clipped to l in {0, 1} (saturation).
        The proof still holds for the non-saturating days (the vast majority).

    Args:
        expert_trading_return_matrix : (T, K).
        maximum_return_bound         : R_max > 0.

    Returns:
        loss_matrix : (T, K) in [0, 1].
    """
    if maximum_return_bound < 1e-12:
        raise ValueError(
            f"maximum_return_bound must be strictly positive; got {maximum_return_bound}"
        )
    raw_mapped_losses = (
        (maximum_return_bound - expert_trading_return_matrix) / (2.0 * maximum_return_bound)
    )
    # Clip to [0, 1]: only activates for tail-event days with |return| > R_max
    return np.clip(raw_mapped_losses, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — AUTOMATED VERIFICATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_all_automated_verification_tests() -> None:
    """
    Run all 5 automated verification tests before the main backtest.

    Tests:
        1. Return-to-loss mapping: argmin(loss) = argmax(return), full rank preserved.
        2. Convexity: combined expert signal stays in [-1, 1].
        3. Look-ahead guardrail: signals at day t unaffected by prices[T].
        4. Transaction cost no-double-counting: net cost <= per-expert sum.
        5. Loss mapping boundary conditions: ±R_max -> {0, 1}; 0 -> 0.5.

    Raises AssertionError immediately on failure. Prints PASS with details.
    """
    print("\n" + "=" * 72)
    print("  AUTOMATED VERIFICATION TESTS")
    print("=" * 72)

    _test1_return_to_loss_mapping_unit_test()
    _test2_convexity_of_combined_expert_signal()
    _test3_look_ahead_guardrail()
    _test4_transaction_cost_no_double_counting()
    _test5_loss_mapping_boundary_conditions()

    print("\n  All 5 verification tests PASSED")
    print("=" * 72)


def _test1_return_to_loss_mapping_unit_test() -> None:
    """
    Test 1: Return-to-loss mapping unit test.

    Hardcoded 5-day, 3-expert example:
        Expert A: large positive returns  -> best performer (lowest loss)
        Expert B: mixed returns           -> middle
        Expert C: large negative returns  -> worst performer (highest loss)

    Verifies both:
        (a) argmin(cumulative loss) == argmax(cumulative return)
        (b) Full ordering is preserved: rank by loss == rank by return (descending)
    """
    # Rows = days, columns = experts
    hardcoded_daily_trading_returns = np.array([
        [ 0.020,  0.005, -0.015],   # Day 0
        [ 0.015, -0.008, -0.020],   # Day 1
        [ 0.018,  0.012, -0.012],   # Day 2
        [-0.003, -0.010, -0.018],   # Day 3
        [ 0.022,  0.003, -0.025],   # Day 4
    ])   # shape (5, 3)

    test_r_max = float(np.max(np.abs(hardcoded_daily_trading_returns)))

    mapped_loss_matrix = map_trading_returns_to_unit_interval_loss_matrix(
        expert_trading_return_matrix=hardcoded_daily_trading_returns,
        maximum_return_bound=test_r_max,
    )

    # Losses must be in [0, 1]
    assert np.all(mapped_loss_matrix >= 0.0), "Mapped loss below 0 detected"
    assert np.all(mapped_loss_matrix <= 1.0), "Mapped loss above 1 detected"

    cumulative_trading_returns = hardcoded_daily_trading_returns.sum(axis=0)
    cumulative_mapped_losses   = mapped_loss_matrix.sum(axis=0)

    # Test (a): argmin(loss) == argmax(return)
    expert_with_minimum_loss    = int(np.argmin(cumulative_mapped_losses))
    expert_with_maximum_return  = int(np.argmax(cumulative_trading_returns))

    assert expert_with_minimum_loss == expert_with_maximum_return, (
        f"TEST 1 FAILED: argmin(loss)={expert_with_minimum_loss} "
        f"!= argmax(return)={expert_with_maximum_return}\n"
        f"  cum_returns={cumulative_trading_returns}\n"
        f"  cum_losses ={cumulative_mapped_losses}"
    )

    # Test (b): full ranking preservation
    ranking_by_loss_ascending   = np.argsort(cumulative_mapped_losses)
    ranking_by_return_descending = np.argsort(-cumulative_trading_returns)

    assert np.array_equal(ranking_by_loss_ascending, ranking_by_return_descending), (
        f"TEST 1 FAILED: Full ranking not preserved.\n"
        f"  Loss rank (asc):    {ranking_by_loss_ascending}\n"
        f"  Return rank (desc): {ranking_by_return_descending}"
    )

    print(
        f"  [PASS] Test 1: Return-to-loss mapping\n"
        f"         argmin(loss) = argmax(return) = Expert {expert_with_maximum_return} (A)\n"
        f"         Full ordering preserved: {ranking_by_return_descending.tolist()}"
    )


def _test2_convexity_of_combined_expert_signal() -> None:
    """
    Test 2: Convexity of the combined expert signal.

    CLAIM: For any w in Delta_K (probability simplex) and s in [-1, 1]^K,
           the portfolio signal Sum_k w_k · s_k in [-1, 1].

    PROOF: Sum_k w_k · (-1) <= Sum_k w_k · s_k <= Sum_k w_k · 1
           ⇒   -1 <= Sum w_k s_k <= +1   ∎

    Verified empirically with 10,000 random (w, s) pairs and K = 36.
    """
    number_of_random_trials = 10_000
    number_of_experts       = TOTAL_NUMBER_OF_EXPERTS   # 36

    # Random weights on the probability simplex (via Dirichlet / exponential trick)
    random_unnormalised_weights = numpy_rng_phase2.exponential(
        scale=1.0, size=(number_of_random_trials, number_of_experts)
    )
    random_probability_weights = (
        random_unnormalised_weights / random_unnormalised_weights.sum(axis=1, keepdims=True)
    )

    # Random signals in [-1, 1]
    random_expert_signals = numpy_rng_phase2.uniform(
        low=-1.0, high=1.0, size=(number_of_random_trials, number_of_experts)
    )

    # Combined portfolio signal: Sum_k w_k · s_k
    combined_portfolio_signals = np.einsum(
        "ij,ij->i", random_probability_weights, random_expert_signals
    )

    assert combined_portfolio_signals.min() >= -1.0 - 1e-10, (
        f"TEST 2 FAILED: Combined signal < -1: {combined_portfolio_signals.min():.8f}"
    )
    assert combined_portfolio_signals.max() <= +1.0 + 1e-10, (
        f"TEST 2 FAILED: Combined signal > +1: {combined_portfolio_signals.max():.8f}"
    )

    print(
        f"  [PASS] Test 2: Convexity of combined signal\n"
        f"         {number_of_random_trials} trials, K={number_of_experts}, "
        f"range=[{combined_portfolio_signals.min():.4f}, "
        f"{combined_portfolio_signals.max():.4f}]"
    )


def _test3_look_ahead_guardrail() -> None:
    """
    Test 3: Look-ahead guardrail.

    Method:
        Construct a synthetic price series of length T+1.  Compute signals
        on the original series. Then create a copy where ONLY prices[T]
        (the last element, which is NEVER read by any signal computation at
        day t <= T-1) is replaced with a large sentinel value (1,000,000).

        If ANY signal at any day t < T changed, it means that signal read
        prices[T] — a look-ahead violation.

        Signal at day t uses prices[price_index=t], i.e., indices 0 … t.
        The maximum price_index accessed across all T days is T-1, not T.
    """
    synthetic_number_of_return_days = 350
    synthetic_price_series = 100.0 + numpy_rng_phase2.normal(
        0, 1, synthetic_number_of_return_days + 1
    ).cumsum()

    signals_from_original_prices = (
        compute_signals_for_single_asset_with_expanding_window_calibration(
            adjusted_close_prices=synthetic_price_series,
            total_number_of_return_days=synthetic_number_of_return_days,
        )
    )

    prices_with_sentinel = synthetic_price_series.copy()
    prices_with_sentinel[synthetic_number_of_return_days] = 1_000_000.0   # sentinel

    signals_from_sentinel_prices = (
        compute_signals_for_single_asset_with_expanding_window_calibration(
            adjusted_close_prices=prices_with_sentinel,
            total_number_of_return_days=synthetic_number_of_return_days,
        )
    )

    for strategy_name in ALL_STRATEGY_TYPES:
        original_signal_array = signals_from_original_prices[strategy_name]
        sentinel_signal_array = signals_from_sentinel_prices[strategy_name]
        valid_comparison_mask = ~(
            np.isnan(original_signal_array) | np.isnan(sentinel_signal_array)
        )
        assert np.allclose(
            original_signal_array[valid_comparison_mask],
            sentinel_signal_array[valid_comparison_mask],
            atol=1e-10,
        ), (
            f"TEST 3 FAILED: Look-ahead contamination detected in strategy "
            f"'{strategy_name}'. Sentinel at prices[T] changed the signals."
        )

    print(
        "  [PASS] Test 3: Look-ahead guardrail\n"
        "         Sentinel at prices[T] did not affect any signal at t < T"
    )


def _test4_transaction_cost_no_double_counting() -> None:
    """
    Test 4: Transaction cost no-double-counting.

    CLAIM: Net portfolio cost <= sum of per-expert costs.

    Equality holds iff no two experts on the same asset move in opposite
    directions. Strict inequality (net < per-expert) holds when offsetting
    moves occur — the net position doesn't change, but naive per-expert
    costs would double-count the turnover.

    Constructed example:
        2 experts on the SAME asset, equal weights (0.5 each).
        Expert 0: position +0.5 -> +0.8  (change = +0.3)
        Expert 1: position +0.5 -> +0.2  (change = -0.3)
        Net position: stays at 0.5 (no net change -> net cost = 0)
        Per-expert cost: 0.5×0.3 + 0.5×0.3 = 0.3 > 0 = net cost
    """
    test_cost_coefficient = 0.001

    weights_at_previous_day = np.array([0.5, 0.5])
    weights_at_current_day  = np.array([0.5, 0.5])

    signals_at_previous_day = np.array([0.5, 0.5])
    signals_at_current_day  = np.array([0.8, 0.2])

    net_position_yesterday = float(np.dot(weights_at_previous_day, signals_at_previous_day))
    net_position_today     = float(np.dot(weights_at_current_day,  signals_at_current_day))

    net_turnover = abs(net_position_today - net_position_yesterday)
    net_cost     = test_cost_coefficient * net_turnover

    per_expert_turnovers    = weights_at_current_day * np.abs(
        signals_at_current_day - signals_at_previous_day
    )
    per_expert_costs_sum    = test_cost_coefficient * float(np.sum(per_expert_turnovers))

    assert net_cost <= per_expert_costs_sum + 1e-12, (
        f"TEST 4 FAILED: Net cost {net_cost:.8f} > per-expert sum {per_expert_costs_sum:.8f}"
    )
    assert abs(net_turnover) < 1e-10, (
        f"TEST 4 FAILED: Expected zero net turnover (cancellation case), "
        f"got {net_turnover:.8f}"
    )

    print(
        f"  [PASS] Test 4: Transaction cost no-double-counting\n"
        f"         net_cost={net_cost:.6f} <= per_expert_sum={per_expert_costs_sum:.6f}\n"
        f"         Cancellation case: opposite signals -> net_turnover={net_turnover:.6f} ~= 0"
    )


def _test5_loss_mapping_boundary_conditions() -> None:
    """
    Test 5: Loss mapping boundary conditions.

    Verifies the three key anchors of the affine rescaling:
        return = +R_max -> loss = 0.0
        return = -R_max -> loss = 1.0
        return =  0     -> loss = 0.5
    """
    test_r_max = 0.05   # arbitrary positive constant

    boundary_test_returns = np.array([[test_r_max, -test_r_max, 0.0]])   # (1, 3)
    boundary_test_losses  = map_trading_returns_to_unit_interval_loss_matrix(
        expert_trading_return_matrix=boundary_test_returns,
        maximum_return_bound=test_r_max,
    )

    assert abs(boundary_test_losses[0, 0] - 0.0) < 1e-10, (
        f"TEST 5 FAILED: +R_max -> loss = {boundary_test_losses[0,0]:.8f} (expected 0.0)"
    )
    assert abs(boundary_test_losses[0, 1] - 1.0) < 1e-10, (
        f"TEST 5 FAILED: -R_max -> loss = {boundary_test_losses[0,1]:.8f} (expected 1.0)"
    )
    assert abs(boundary_test_losses[0, 2] - 0.5) < 1e-10, (
        f"TEST 5 FAILED: 0 return -> loss = {boundary_test_losses[0,2]:.8f} (expected 0.5)"
    )

    print(
        f"  [PASS] Test 5: Loss mapping boundary conditions\n"
        f"         +R_max -> {boundary_test_losses[0,0]:.1f}, "
        f"         -R_max -> {boundary_test_losses[0,1]:.1f}, "
        f"         0 -> {boundary_test_losses[0,2]:.1f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TRANSACTION COST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_transaction_costs_on_net_positions(
    expert_weight_history: np.ndarray,
    expert_signal_matrix: np.ndarray,
    transaction_cost_coefficient: float,
) -> np.ndarray:
    """
    Compute daily transaction costs from NET portfolio position changes.

    DESIGN NOTE:
    Transaction costs are applied POST-HOC: the Hedge/AdaHedge weight update
    is computed from the raw (pre-cost) loss matrix, so the algorithm has NO
    incentive to reduce turnover even when switching costs are high.

    This is a standard and legitimate simplification.  Cost-aware online
    learning — where the learner sees costs and optimises for them — falls
    under the theory of competitive analysis with switching costs, a separate
    problem class NOT covered by the Hedge regret bounds we are working with
    (Theorem 3 of van Erven et al. 2011).  Adding costs to the evaluation
    (but not to the update) gives an honest picture of real-world performance
    without over-complicating the theoretical framework.

    Cost model:
        net_position_t^a = Sum_{e : asset(e)=a} weight_t^e × signal_t^e
        daily_cost_t     = c × Sum_a |net_position_t^a - net_position_{t-1}^a|

    Args:
        expert_weight_history         : (T+1, K) — weight_history[t] used for day t.
        expert_signal_matrix          : (T, K)   — signal[t] for day t.
        transaction_cost_coefficient  : c >= 0.

    Returns:
        daily_transaction_cost_fractions : (T,) — cost as fraction of portfolio value.
    """
    total_number_of_return_days = expert_signal_matrix.shape[0]

    # Compute net position per asset per day: shape (T, N_assets)
    net_position_matrix_by_day_and_asset = np.zeros(
        (total_number_of_return_days, NUMBER_OF_ASSETS_IN_BASKET), dtype=np.float64
    )
    for asset_index in range(NUMBER_OF_ASSETS_IN_BASKET):
        expert_mask_for_this_asset = (EXPERT_ASSET_INDEX_ARRAY == asset_index)
        # Net position = weighted sum of expert signals for this asset
        net_position_matrix_by_day_and_asset[:, asset_index] = (
            expert_weight_history[:total_number_of_return_days, expert_mask_for_this_asset]
            * expert_signal_matrix[:, expert_mask_for_this_asset]
        ).sum(axis=1)

    # Daily position change (prepend zeros = flat position before backtest starts).
    #
    # MODEL ASSUMPTION NOTE (portfolio drift):
    # Turnover is computed as |target_today - target_yesterday|, i.e., the
    # difference between two consecutive *target* net positions. This implicitly
    # assumes end-of-day execution at the closing price: the portfolio is always
    # exactly at its target weight at the start of each day, so no intra-day
    # drift correction is needed. Under this assumption, a strategy like Uniform
    # CRP with constant signals correctly shows zero turnover on unchanged-signal
    # days (no rebalancing needed if we hit yesterday's target exactly).
    #
    # A more conservative "implementation shortfall" model would apply today's
    # asset returns to yesterday's positions first (portfolio drift), then
    # compute the gap to today's target. That model is standard in execution-cost
    # research but is beyond the scope of this online-learning evaluation, where
    # we follow the convention used in the majority of academic strategy backtests.
    previous_day_net_positions  = np.zeros(NUMBER_OF_ASSETS_IN_BASKET, dtype=np.float64)
    daily_transaction_costs     = np.zeros(total_number_of_return_days, dtype=np.float64)

    for day_index in range(total_number_of_return_days):
        current_day_net_positions   = net_position_matrix_by_day_and_asset[day_index]
        absolute_position_changes   = np.abs(current_day_net_positions - previous_day_net_positions)
        daily_transaction_costs[day_index] = (
            transaction_cost_coefficient * float(np.sum(absolute_position_changes))
        )
        previous_day_net_positions = current_day_net_positions.copy()

    return daily_transaction_costs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — BACKTEST ENGINE (ALL STRATEGIES)
# ══════════════════════════════════════════════════════════════════════════════

def _compound_daily_returns_to_wealth_path(
    daily_gross_portfolio_return_rates: np.ndarray,
    daily_transaction_cost_fractions: np.ndarray,
) -> np.ndarray:
    """
    Compound daily net returns into a cumulative wealth path.

    W_t = W_0 × Π_{s=0}^{t-1} max(1 + gross_return_s - cost_s, 0.01)

    The floor of 0.01 prevents the wealth from going negative in extreme
    scenarios (e.g., full leverage + tail event + high costs).

    Returns:
        cumulative_wealth_path : (T,) — portfolio value in dollars at end of each day.
    """
    net_daily_return_rates = daily_gross_portfolio_return_rates - daily_transaction_cost_fractions
    gross_daily_factors    = np.maximum(1.0 + net_daily_return_rates, 0.01)
    cumulative_wealth_path = (
        INITIAL_PORTFOLIO_WEALTH_DOLLARS * np.cumprod(gross_daily_factors)
    )
    return cumulative_wealth_path


def run_adahedge_trading_strategy(
    expert_signal_matrix: np.ndarray,
    expert_trading_return_matrix: np.ndarray,
    loss_matrix: np.ndarray,
    transaction_cost_coefficient: float = TRANSACTION_COST_BASE_COEFFICIENT,
    segment_learning_rate_decay_factor: float = 2.0,
) -> dict:
    """
    Run the full AdaHedge trading strategy (weights -> portfolio returns -> wealth).

    Step 1: Run Phase 1 AdaHedge on the loss matrix to obtain per-day weights.
    Step 2: Compute gross portfolio returns using those weights.
    Step 3: Compute transaction costs on net position changes (post-hoc).
    Step 4: Compound net returns into a wealth path.
    """
    total_number_of_return_days = loss_matrix.shape[0]

    adahedge_results = run_adahedge_algorithm(
        loss_matrix=loss_matrix,
        segment_learning_rate_decay_factor=segment_learning_rate_decay_factor,
    )
    weight_history = adahedge_results["expert_probability_allocation_weights"]   # (T+1, K)

    # Gross portfolio return = Sum_e weight_t^e × tradingReturn_t^e
    daily_gross_portfolio_return_rates = np.array([
        float(np.dot(weight_history[day_idx], expert_trading_return_matrix[day_idx]))
        for day_idx in range(total_number_of_return_days)
    ])

    daily_transaction_cost_fractions = compute_daily_transaction_costs_on_net_positions(
        expert_weight_history=weight_history,
        expert_signal_matrix=expert_signal_matrix,
        transaction_cost_coefficient=transaction_cost_coefficient,
    )

    cumulative_wealth_path = _compound_daily_returns_to_wealth_path(
        daily_gross_portfolio_return_rates=daily_gross_portfolio_return_rates,
        daily_transaction_cost_fractions=daily_transaction_cost_fractions,
    )

    return {
        "strategy_display_name":                    "AdaHedge (phi=2)",
        "cumulative_wealth_path":                   cumulative_wealth_path,
        "daily_gross_portfolio_return_rates":       daily_gross_portfolio_return_rates,
        "daily_transaction_cost_fractions":         daily_transaction_cost_fractions,
        "cumulative_regret_of_learner":             adahedge_results["cumulative_regret_of_learner"],
        "cumulative_loss_of_learner":               adahedge_results["cumulative_loss_of_learner"],
        "expert_probability_allocation_weights":    weight_history,
        "learning_rate_at_each_round":              adahedge_results["learning_rate_at_each_round"],
        "segment_boundary_round_indices":           adahedge_results["segment_boundary_round_indices"],
        "total_number_of_segments_used":            adahedge_results["total_number_of_segments_used"],
    }


def run_hedge_fixed_rate_trading_strategy(
    expert_signal_matrix: np.ndarray,
    expert_trading_return_matrix: np.ndarray,
    loss_matrix: np.ndarray,
    learning_rate_eta: float,
    strategy_display_name: str,
    transaction_cost_coefficient: float = TRANSACTION_COST_BASE_COEFFICIENT,
) -> dict:
    """Run fixed-eta Hedge and compute the portfolio wealth path."""
    total_number_of_return_days = loss_matrix.shape[0]

    hedge_results  = run_hedge_fixed_learning_rate(
        loss_matrix=loss_matrix, learning_rate=learning_rate_eta
    )
    weight_history = hedge_results["expert_probability_allocation_weights"]   # (T+1, K)

    daily_gross_portfolio_return_rates = np.array([
        float(np.dot(weight_history[day_idx], expert_trading_return_matrix[day_idx]))
        for day_idx in range(total_number_of_return_days)
    ])

    daily_transaction_cost_fractions = compute_daily_transaction_costs_on_net_positions(
        expert_weight_history=weight_history,
        expert_signal_matrix=expert_signal_matrix,
        transaction_cost_coefficient=transaction_cost_coefficient,
    )

    cumulative_wealth_path = _compound_daily_returns_to_wealth_path(
        daily_gross_portfolio_return_rates=daily_gross_portfolio_return_rates,
        daily_transaction_cost_fractions=daily_transaction_cost_fractions,
    )

    return {
        "strategy_display_name":                strategy_display_name,
        "cumulative_wealth_path":               cumulative_wealth_path,
        "daily_gross_portfolio_return_rates":   daily_gross_portfolio_return_rates,
        "daily_transaction_cost_fractions":     daily_transaction_cost_fractions,
        "cumulative_regret_of_learner":         hedge_results["cumulative_regret_of_learner"],
        "cumulative_loss_of_learner":           hedge_results["cumulative_loss_of_learner"],
        "expert_probability_allocation_weights": weight_history,
    }


def run_buy_and_hold_equal_weight_benchmark(
    asset_daily_return_matrix: np.ndarray,
    transaction_cost_coefficient: float = TRANSACTION_COST_BASE_COEFFICIENT,
) -> dict:
    """
    True Buy-and-Hold Equal-Weight benchmark.

    Initial capital split equally across all N_assets ETFs. Positions are
    never rebalanced — asset values drift freely with market returns.

    Portfolio return at day t reflects the drifted (market-cap-like) weights,
    NOT the constant 1/N equal weight. This is computed by tracking the
    dollar value in each ETF explicitly.

    Transaction cost: applied only once on day 0 (initial entry). The initial
    total turnover is Sum_a |1/N - 0| = 1.0, so cost₀ = c × 1.0 = c.
    """
    total_number_of_return_days = asset_daily_return_matrix.shape[0]
    initial_value_per_etf = INITIAL_PORTFOLIO_WEALTH_DOLLARS / NUMBER_OF_ASSETS_IN_BASKET

    # Track dollar value in each ETF day by day
    asset_values_over_time = np.zeros(
        (total_number_of_return_days + 1, NUMBER_OF_ASSETS_IN_BASKET), dtype=np.float64
    )
    asset_values_over_time[0] = initial_value_per_etf   # initial allocation

    for day_index in range(total_number_of_return_days):
        asset_values_over_time[day_index + 1] = (
            asset_values_over_time[day_index] * (1.0 + asset_daily_return_matrix[day_index])
        )

    total_portfolio_value_over_time = asset_values_over_time.sum(axis=1)   # (T+1,)
    cumulative_wealth_path = total_portfolio_value_over_time[1:]            # (T,)

    # Daily gross returns from the compounding portfolio value
    daily_gross_portfolio_return_rates = (
        (total_portfolio_value_over_time[1:] - total_portfolio_value_over_time[:-1])
        / total_portfolio_value_over_time[:-1]
    )

    # Transaction costs: one-time entry cost on day 0, zero thereafter
    daily_transaction_cost_fractions        = np.zeros(total_number_of_return_days, dtype=np.float64)
    initial_entry_turnover                  = 1.0   # going from 0 to 1/N × N = 1 in total
    daily_transaction_cost_fractions[0]     = transaction_cost_coefficient * initial_entry_turnover

    # Recompute wealth with the day-0 cost applied
    cumulative_wealth_path = _compound_daily_returns_to_wealth_path(
        daily_gross_portfolio_return_rates=daily_gross_portfolio_return_rates,
        daily_transaction_cost_fractions=daily_transaction_cost_fractions,
    )

    return {
        "strategy_display_name":              "Buy-and-Hold EW",
        "cumulative_wealth_path":             cumulative_wealth_path,
        "daily_gross_portfolio_return_rates": daily_gross_portfolio_return_rates,
        "daily_transaction_cost_fractions":   daily_transaction_cost_fractions,
    }


def run_uniform_constant_rebalanced_portfolio_benchmark(
    expert_signal_matrix: np.ndarray,
    expert_trading_return_matrix: np.ndarray,
    transaction_cost_coefficient: float = TRANSACTION_COST_BASE_COEFFICIENT,
) -> dict:
    """
    Uniform Constant Rebalanced Portfolio (CRP) benchmark.

    Holds equal weight 1/K on all 36 experts, rebalancing back to 1/K daily.
    Equivalent to Hedge with eta -> 0 (no learning from losses).

    Daily return = arithmetic mean of all expert trading returns.
    Transaction costs are computed on net positions as positions change
    due to daily signal drift (even though weights stay constant at 1/K).
    """
    total_number_of_return_days = expert_trading_return_matrix.shape[0]

    # Weight history: constant 1/K for all days (T+1, K)
    uniform_weight_fraction = 1.0 / TOTAL_NUMBER_OF_EXPERTS
    uniform_weight_history  = np.full(
        (total_number_of_return_days + 1, TOTAL_NUMBER_OF_EXPERTS),
        uniform_weight_fraction, dtype=np.float64
    )

    # Gross returns: simple average across all expert trading returns
    daily_gross_portfolio_return_rates = expert_trading_return_matrix.mean(axis=1)

    daily_transaction_cost_fractions = compute_daily_transaction_costs_on_net_positions(
        expert_weight_history=uniform_weight_history,
        expert_signal_matrix=expert_signal_matrix,
        transaction_cost_coefficient=transaction_cost_coefficient,
    )

    cumulative_wealth_path = _compound_daily_returns_to_wealth_path(
        daily_gross_portfolio_return_rates=daily_gross_portfolio_return_rates,
        daily_transaction_cost_fractions=daily_transaction_cost_fractions,
    )

    return {
        "strategy_display_name":                  "Uniform CRP",
        "cumulative_wealth_path":                 cumulative_wealth_path,
        "daily_gross_portfolio_return_rates":     daily_gross_portfolio_return_rates,
        "daily_transaction_cost_fractions":       daily_transaction_cost_fractions,
        "expert_probability_allocation_weights":  uniform_weight_history,
    }


def run_best_expert_in_hindsight_benchmark(
    expert_signal_matrix: np.ndarray,
    expert_trading_return_matrix: np.ndarray,
    transaction_cost_coefficient: float = TRANSACTION_COST_BASE_COEFFICIENT,
) -> dict:
    """
    Best Expert in Hindsight benchmark.

    Identifies the single (strategy × asset) expert with the highest
    total cumulative return over the entire backtest period (using full
    hindsight) and simulates running that expert alone.  This is the
    oracle comparator that online learning algorithms aim to compete with.

    Transaction costs for a single expert: c × |signal_t - signal_{t-1}|
    (signal changes directly drive position changes).
    """
    cumulative_return_per_expert    = expert_trading_return_matrix.sum(axis=0)
    best_expert_index               = int(np.argmax(cumulative_return_per_expert))
    best_expert_label               = EXPERT_LABEL_ARRAY[best_expert_index]

    print(
        f"\n  [HINDSIGHT] Best expert: {best_expert_label}\n"
        f"                Cum. return: {cumulative_return_per_expert[best_expert_index]*100:.2f}%"
    )

    best_expert_daily_returns = expert_trading_return_matrix[:, best_expert_index]
    best_expert_daily_signals = expert_signal_matrix[:, best_expert_index]

    # Signal changes drive position turnover (single expert, no netting)
    signal_absolute_changes             = np.abs(np.diff(best_expert_daily_signals, prepend=0.0))
    daily_transaction_cost_fractions    = transaction_cost_coefficient * signal_absolute_changes

    cumulative_wealth_path = _compound_daily_returns_to_wealth_path(
        daily_gross_portfolio_return_rates=best_expert_daily_returns,
        daily_transaction_cost_fractions=daily_transaction_cost_fractions,
    )

    return {
        "strategy_display_name":              "Best Expert (Hindsight)",
        "cumulative_wealth_path":             cumulative_wealth_path,
        "daily_gross_portfolio_return_rates": best_expert_daily_returns,
        "daily_transaction_cost_fractions":   daily_transaction_cost_fractions,
        "best_expert_index":                  best_expert_index,
        "best_expert_label":                  best_expert_label,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_performance_metrics_for_strategy(
    cumulative_wealth_path: np.ndarray,
    daily_gross_return_rates: np.ndarray,
    daily_cost_fractions: np.ndarray,
) -> dict:
    """
    Compute the full suite of portfolio performance metrics.

    Returns a dict with 10 named scalar metrics.

    Note on Sharpe:  risk-free rate = 0 (global DAILY_RISK_FREE_RATE constant).
    All strategies are affected equally; relative rankings are preserved.

    Note on annualised return for Sharpe:
    The Sharpe ratio strictly requires the ARITHMETIC mean of returns in the
    numerator (E[r] / sigma). Using CAGR (geometric mean) instead would
    penalise volatile strategies twice -- once because CAGR is dragged down by
    variance relative to the arithmetic mean (CAGR ~= E[r] - sigma^2/2), and
    again through the denominator sigma. This function therefore uses the
    arithmetic mean * 252 for the Sharpe numerator. CAGR is reported separately
    as 'annualised_cagr_fraction' for use in the wealth-path comparison table.
    Both the rolling Sharpe (compute_rolling_annualised_sharpe_ratio) and this
    summary metric now use arithmetic mean consistently.
    """
    total_trading_days      = len(cumulative_wealth_path)
    net_daily_return_rates  = daily_gross_return_rates - daily_cost_fractions
    number_of_years         = total_trading_days / TRADING_DAYS_PER_YEAR

    # Annualised return — two variants:
    #   CAGR (geometric): correct for reporting long-run compounded wealth growth.
    #   Arithmetic mean * 252: correct for the Sharpe ratio numerator.
    total_return_fraction                  = cumulative_wealth_path[-1] / INITIAL_PORTFOLIO_WEALTH_DOLLARS - 1.0
    annualised_cagr_fraction               = (1.0 + total_return_fraction) ** (1.0 / number_of_years) - 1.0
    annualised_arithmetic_return_fraction  = float(np.mean(net_daily_return_rates)) * TRADING_DAYS_PER_YEAR

    # Annualised volatility
    annualised_volatility_fraction = float(np.std(net_daily_return_rates)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio (risk-free = 0): uses arithmetic mean, not CAGR
    if annualised_volatility_fraction > 1e-10:
        sharpe_ratio = (
            (annualised_arithmetic_return_fraction - ANNUAL_RISK_FREE_RATE)
            / annualised_volatility_fraction
        )
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    rolling_peak_values         = np.maximum.accumulate(cumulative_wealth_path)
    drawdown_series             = (cumulative_wealth_path - rolling_peak_values) / rolling_peak_values
    maximum_drawdown_fraction   = float(drawdown_series.min())

    # Calmar ratio: uses CAGR (long-run growth rate vs worst peak-to-trough)
    calmar_ratio = (
        annualised_cagr_fraction / abs(maximum_drawdown_fraction)
        if abs(maximum_drawdown_fraction) > 1e-10
        else np.nan
    )

    # Average daily cost and annualised cost drag
    average_daily_cost_fraction         = float(np.mean(daily_cost_fractions))
    annualised_cost_drag_fraction       = average_daily_cost_fraction * TRADING_DAYS_PER_YEAR

    return {
        "total_return_fraction":                   total_return_fraction,
        "annualised_cagr_fraction":                annualised_cagr_fraction,
        "annualised_arithmetic_return_fraction":   annualised_arithmetic_return_fraction,
        "annualised_volatility_fraction":          annualised_volatility_fraction,
        "sharpe_ratio_risk_free_zero":             sharpe_ratio,
        "maximum_drawdown_fraction":               maximum_drawdown_fraction,
        "calmar_ratio":                            calmar_ratio,
        "average_daily_cost_fraction":             average_daily_cost_fraction,
        "annualised_cost_drag_fraction":           annualised_cost_drag_fraction,
        "total_trading_days":                      total_trading_days,
        "number_of_years":                         number_of_years,
    }


def compute_rolling_annualised_sharpe_ratio(
    daily_gross_return_rates: np.ndarray,
    daily_cost_fractions: np.ndarray,
    rolling_window_size: int = ROLLING_SHARPE_WINDOW_SIZE_IN_DAYS,
) -> np.ndarray:
    """
    Compute the rolling annualised Sharpe ratio over a sliding window.

    Returns NaN for the first (rolling_window_size - 1) days.
    """
    net_daily_returns          = daily_gross_return_rates - daily_cost_fractions
    total_days                 = len(net_daily_returns)
    rolling_sharpe_series      = np.full(total_days, np.nan)

    for day_index in range(rolling_window_size - 1, total_days):
        window_net_returns     = net_daily_returns[day_index - rolling_window_size + 1 : day_index + 1]
        window_excess_returns  = window_net_returns - DAILY_RISK_FREE_RATE
        window_mean            = float(np.mean(window_excess_returns))
        window_std             = float(np.std(window_excess_returns))
        if window_std > 1e-10:
            rolling_sharpe_series[day_index] = (
                (window_mean / window_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
            )

    return rolling_sharpe_series


def compute_drawdown_series_from_wealth_path(
    cumulative_wealth_path: np.ndarray,
) -> np.ndarray:
    """Compute the drawdown at each day as fraction below rolling peak."""
    rolling_peak = np.maximum.accumulate(cumulative_wealth_path)
    return (cumulative_wealth_path - rolling_peak) / rolling_peak


def compute_herfindahl_weight_concentration_index(
    expert_weight_history: np.ndarray,
) -> np.ndarray:
    """
    Compute the Herfindahl–Hirschman Index (HHI) per day.

    HHI = Sum_k w_k^2 in [1/K, 1].
        HHI = 1/K -> perfectly uniform (maximally diversified).
        HHI = 1.0 -> all weight on one expert (fully concentrated).

    Returns:
        hhi_series : (T,) — one scalar per trading day.
    """
    # weight_history has shape (T+1, K); use rows 0..T-1 (trading-day weights)
    trading_day_weights = expert_weight_history[:-1]   # (T, K)
    herfindahl_index_per_day = np.sum(trading_day_weights ** 2, axis=1)
    return herfindahl_index_per_day


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_wealth_comparison(
    all_strategy_results: dict,
    return_day_date_index: pd.DatetimeIndex,
) -> None:
    """
    Plot 1: Cumulative wealth comparison for all 7 strategies and benchmarks.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    for strategy_name, result in all_strategy_results.items():
        wealth  = result["cumulative_wealth_path"]
        color   = BENCHMARK_COLOR_MAP.get(strategy_name, "#FFFFFF")
        is_hero = "AdaHedge" in strategy_name

        ax.plot(
            return_day_date_index, wealth,
            color=color,
            linewidth=2.6 if is_hero else 1.4,
            alpha=1.0 if is_hero else 0.72,
            zorder=10 if is_hero else 3,
            label=strategy_name,
        )

    ax.axhline(
        y=INITIAL_PORTFOLIO_WEALTH_DOLLARS, color=_SPINE,
        linewidth=0.9, linestyle="--", alpha=0.5, label="Initial capital"
    )
    ax.legend(loc="upper left", fontsize=8, borderpad=0.6, labelspacing=0.4)
    _apply_dark_axis_styling(
        ax,
        title="Cumulative Portfolio Wealth — US Sector ETFs (2019–2026)",
        xlabel="Date",
        ylabel=f"Wealth (USD, starting from ${INITIAL_PORTFOLIO_WEALTH_DOLLARS:,.0f})",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    _save_plot_to_directory(fig, "trading_01_cumulative_wealth")


def plot_weight_evolution_heatmaps(
    adahedge_weight_history: np.ndarray,
    return_day_date_index: pd.DatetimeIndex,
) -> None:
    """
    Plot 2: AdaHedge weight evolution — three-panel heatmap.

    Panels:
        Top (full):          36-expert weight matrix  (T × 36)
        Bottom-left:         Strategy marginal        (T × 4)  [sum by strategy]
        Bottom-right:        Asset marginal           (T × 9)  [sum by ETF]

    The marginals make the full heatmap legible at high density: while the
    36-expert matrix shows the fine-grained expert competition, the marginals
    reveal which overall strategies and sectors AdaHedge is tilting toward.
    """
    total_return_days = adahedge_weight_history.shape[0] - 1
    weights           = adahedge_weight_history[:total_return_days]   # (T, 36)

    # Compute strategy marginal: sum weights over all assets for each strategy
    strategy_marginal_weights = np.zeros((total_return_days, NUMBER_OF_STRATEGY_TYPES))
    for strategy_index in range(NUMBER_OF_STRATEGY_TYPES):
        strategy_mask = (EXPERT_STRATEGY_INDEX_ARRAY == strategy_index)
        strategy_marginal_weights[:, strategy_index] = weights[:, strategy_mask].sum(axis=1)

    # Compute asset marginal: sum weights over all strategies for each asset
    asset_marginal_weights = np.zeros((total_return_days, NUMBER_OF_ASSETS_IN_BASKET))
    for asset_index in range(NUMBER_OF_ASSETS_IN_BASKET):
        asset_mask = (EXPERT_ASSET_INDEX_ARRAY == asset_index)
        asset_marginal_weights[:, asset_index] = weights[:, asset_mask].sum(axis=1)

    # Custom colormap: dark navy -> vivid magenta (matching hero colour)
    adahedge_heatmap_colormap = LinearSegmentedColormap.from_list(
        "adahedge_heatmap", [_PANEL, "#7B2D8B", COLOR_ADAHEDGE], N=256
    )

    # Select x-axis tick positions (about 12 evenly spaced date labels)
    date_tick_indices = np.linspace(0, total_return_days - 1, 12, dtype=int)
    date_tick_labels  = [
        return_day_date_index[i].strftime("%Y-%m") for i in date_tick_indices
    ]

    # Short expert labels for the y-axis of the full heatmap
    expert_short_labels = [
        f"{ALL_STRATEGY_TYPES[si].split('_')[0][:3]}|{SECTOR_ETF_TICKERS[ai]}"
        for si in range(NUMBER_OF_STRATEGY_TYPES)
        for ai in range(NUMBER_OF_ASSETS_IN_BASKET)
    ]

    fig = plt.figure(figsize=(18, 13))
    grid_spec = gridspec.GridSpec(
        2, 2, height_ratios=[2.2, 1.0], hspace=0.38, wspace=0.30,
        figure=fig
    )

    # ── Panel A: full 36-expert heatmap ───────────────────────────────────────
    ax_full = fig.add_subplot(grid_spec[0, :])   # spans both columns
    max_weight_for_scale = max(weights.max(), 1.0 / TOTAL_NUMBER_OF_EXPERTS * 3)

    image_full = ax_full.imshow(
        weights.T, aspect="auto",
        cmap=adahedge_heatmap_colormap,
        vmin=0, vmax=max_weight_for_scale,
        interpolation="nearest",
    )
    ax_full.set_yticks(range(TOTAL_NUMBER_OF_EXPERTS))
    ax_full.set_yticklabels(expert_short_labels, fontsize=5.5)
    ax_full.set_xticks(date_tick_indices)
    ax_full.set_xticklabels(date_tick_labels, rotation=40, ha="right", fontsize=7)
    ax_full.set_title("AdaHedge Expert Weights — All 36 Experts", fontweight="bold", pad=8)
    ax_full.set_xlabel("Date", labelpad=5)
    ax_full.set_ylabel("Expert (Strategy | ETF)", labelpad=5)
    fig.colorbar(image_full, ax=ax_full, fraction=0.008, pad=0.01, label="Weight w_t^e")

    # ── Panel B: strategy marginal ─────────────────────────────────────────────
    ax_strategy = fig.add_subplot(grid_spec[1, 0])
    strategy_short_labels = [s.replace("_", " ") for s in ALL_STRATEGY_TYPES]

    image_strategy = ax_strategy.imshow(
        strategy_marginal_weights.T, aspect="auto",
        cmap=adahedge_heatmap_colormap, vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax_strategy.set_yticks(range(NUMBER_OF_STRATEGY_TYPES))
    ax_strategy.set_yticklabels(strategy_short_labels, fontsize=8)
    ax_strategy.set_xticks(date_tick_indices)
    ax_strategy.set_xticklabels(date_tick_labels, rotation=40, ha="right", fontsize=7)
    ax_strategy.set_title("Strategy Marginal  (Sum weight by strategy)", fontweight="bold", pad=6)
    ax_strategy.set_xlabel("Date", labelpad=5)
    ax_strategy.set_ylabel("Strategy", labelpad=5)
    fig.colorbar(image_strategy, ax=ax_strategy, fraction=0.035, pad=0.02, label="Sum weight")

    # ── Panel C: asset (ETF) marginal ─────────────────────────────────────────
    ax_asset = fig.add_subplot(grid_spec[1, 1])

    image_asset = ax_asset.imshow(
        asset_marginal_weights.T, aspect="auto",
        cmap=adahedge_heatmap_colormap, vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax_asset.set_yticks(range(NUMBER_OF_ASSETS_IN_BASKET))
    ax_asset.set_yticklabels(SECTOR_ETF_TICKERS, fontsize=8)
    ax_asset.set_xticks(date_tick_indices)
    ax_asset.set_xticklabels(date_tick_labels, rotation=40, ha="right", fontsize=7)
    ax_asset.set_title("Asset Marginal  (Sum weight by ETF)", fontweight="bold", pad=6)
    ax_asset.set_xlabel("Date", labelpad=5)
    ax_asset.set_ylabel("ETF", labelpad=5)
    fig.colorbar(image_asset, ax=ax_asset, fraction=0.035, pad=0.02, label="Sum weight")

    fig.suptitle(
        "AdaHedge Weight Evolution Heatmaps — Full + Strategy + Asset Marginals",
        fontsize=13, fontweight="bold", color=_TITLE, y=1.005,
    )
    _save_plot_to_directory(fig, "trading_02_weight_heatmaps")


def plot_net_asset_positions_overlaid_on_prices(
    adahedge_weight_history: np.ndarray,
    expert_signal_matrix: np.ndarray,
    price_matrix: np.ndarray,
    return_day_date_index: pd.DatetimeIndex,
) -> None:
    """
    Plot 3: AdaHedge net asset positions overlaid on normalised ETF prices.

    For each of the 9 ETFs, shows:
        Left y-axis:  normalised price (price / price on first return day = 1.0)
        Right y-axis: net portfolio position (green fill = long, blue fill = short)
    """
    total_return_days       = expert_signal_matrix.shape[0]
    weights_for_return_days = adahedge_weight_history[:total_return_days]   # (T, 36)

    # Net position per asset: (T, N_assets)
    net_position_per_asset_over_time = np.zeros(
        (total_return_days, NUMBER_OF_ASSETS_IN_BASKET), dtype=np.float64
    )
    for asset_index in range(NUMBER_OF_ASSETS_IN_BASKET):
        asset_expert_mask = (EXPERT_ASSET_INDEX_ARRAY == asset_index)
        net_position_per_asset_over_time[:, asset_index] = (
            weights_for_return_days[:, asset_expert_mask]
            * expert_signal_matrix[:, asset_expert_mask]
        ).sum(axis=1)

    # Normalise prices to 1.0 on the first return day (row 1 of price_matrix)
    first_day_prices    = price_matrix[1, :]           # shape (N_assets,)
    normalised_prices   = price_matrix[1:] / first_day_prices[np.newaxis, :]  # (T, N_assets)

    number_of_subplot_columns = 3
    number_of_subplot_rows    = (NUMBER_OF_ASSETS_IN_BASKET + number_of_subplot_columns - 1) \
                                 // number_of_subplot_columns

    fig, subplot_axes = plt.subplots(
        number_of_subplot_rows, number_of_subplot_columns,
        figsize=(16, 4.5 * number_of_subplot_rows),
    )
    subplot_axes_flat = subplot_axes.ravel()

    for asset_index, ticker_symbol in enumerate(SECTOR_ETF_TICKERS):
        ax_price    = subplot_axes_flat[asset_index]
        ax_position = ax_price.twinx()

        # ETF normalised price
        ax_price.plot(
            return_day_date_index,
            normalised_prices[:, asset_index],
            color=COLOR_BUYHOLD_EW, linewidth=1.3, alpha=0.85, label="Norm. Price",
        )

        # Net position: positive (long) and negative (short) as filled areas
        asset_net_positions = net_position_per_asset_over_time[:, asset_index]

        ax_position.fill_between(
            return_day_date_index, 0, asset_net_positions,
            where=(asset_net_positions >= 0),
            color=COLOR_ADAHEDGE, alpha=0.38, label="Long (net > 0)",
        )
        ax_position.fill_between(
            return_day_date_index, 0, asset_net_positions,
            where=(asset_net_positions < 0),
            color=COLOR_HEDGE_WORST, alpha=0.38, label="Short (net < 0)",
        )
        ax_position.axhline(y=0, color=_SPINE, linewidth=0.7, linestyle="--")
        ax_position.set_ylim(-1.05, 1.05)
        ax_position.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_position.tick_params(axis="y", labelsize=7, colors=COLOR_ADAHEDGE)
        ax_position.set_ylabel("Net Position", color=COLOR_ADAHEDGE, fontsize=7, labelpad=4)

        ax_price.set_title(
            f"{ticker_symbol} — {SECTOR_ETF_DISPLAY_NAMES[ticker_symbol]}",
            fontweight="bold", fontsize=9,
        )
        ax_price.set_xlabel("Date", fontsize=7, labelpad=4)
        ax_price.set_ylabel("Norm. Price", color=COLOR_BUYHOLD_EW, fontsize=7, labelpad=4)
        ax_price.tick_params(axis="x", rotation=30, labelsize=7)
        ax_price.tick_params(axis="y", labelsize=7, colors=COLOR_BUYHOLD_EW)
        ax_price.spines["top"].set_visible(False)
        ax_price.grid(True, linewidth=0.4, alpha=0.7)

    # Hide unused subplot cells
    for unused_idx in range(NUMBER_OF_ASSETS_IN_BASKET, len(subplot_axes_flat)):
        subplot_axes_flat[unused_idx].set_visible(False)

    fig.suptitle(
        "AdaHedge Net Asset Positions Overlaid on Normalised ETF Prices",
        fontsize=12, fontweight="bold", color=_TITLE, y=1.01,
    )
    fig.tight_layout()
    _save_plot_to_directory(fig, "trading_03_net_positions")


def plot_rolling_sharpe_ratio_comparison(
    all_strategy_results: dict,
    return_day_date_index: pd.DatetimeIndex,
) -> None:
    """
    Plot 4: Rolling 60-day Sharpe ratio comparison for all strategies.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(y=0, color=_SPINE, linewidth=0.8, linestyle="--", alpha=0.5)

    for strategy_name, result in all_strategy_results.items():
        rolling_sharpe  = compute_rolling_annualised_sharpe_ratio(
            daily_gross_return_rates=result["daily_gross_portfolio_return_rates"],
            daily_cost_fractions=result["daily_transaction_cost_fractions"],
        )
        color   = BENCHMARK_COLOR_MAP.get(strategy_name, "#FFFFFF")
        is_hero = "AdaHedge" in strategy_name

        ax.plot(
            return_day_date_index, rolling_sharpe,
            color=color,
            linewidth=2.4 if is_hero else 1.2,
            alpha=1.0 if is_hero else 0.65,
            zorder=10 if is_hero else 3,
            label=strategy_name,
        )

    ax.legend(loc="lower right", fontsize=8, borderpad=0.6, labelspacing=0.4)
    _apply_dark_axis_styling(
        ax,
        title=(
            f"Rolling {ROLLING_SHARPE_WINDOW_SIZE_IN_DAYS}-Day Sharpe Ratio "
            "(Risk-Free Rate = 0)"
        ),
        xlabel="Date",
        ylabel=f"Annualised Sharpe (rolling {ROLLING_SHARPE_WINDOW_SIZE_IN_DAYS}d)",
    )
    fig.tight_layout()
    _save_plot_to_directory(fig, "trading_04_rolling_sharpe")


def plot_drawdown_comparison(
    all_strategy_results: dict,
    return_day_date_index: pd.DatetimeIndex,
) -> None:
    """
    Plot 5: Portfolio drawdown over time for all strategies.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(y=0, color=_SPINE, linewidth=0.7, alpha=0.4)

    for strategy_name, result in all_strategy_results.items():
        drawdown_series = compute_drawdown_series_from_wealth_path(
            result["cumulative_wealth_path"]
        ) * 100.0   # convert to percent

        color   = BENCHMARK_COLOR_MAP.get(strategy_name, "#FFFFFF")
        is_hero = "AdaHedge" in strategy_name

        ax.plot(
            return_day_date_index, drawdown_series,
            color=color,
            linewidth=2.4 if is_hero else 1.2,
            alpha=1.0 if is_hero else 0.65,
            zorder=10 if is_hero else 3,
            label=strategy_name,
        )

    ax.legend(loc="lower left", fontsize=8, borderpad=0.6, labelspacing=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    _apply_dark_axis_styling(
        ax, title="Portfolio Drawdown Over Time",
        xlabel="Date", ylabel="Drawdown (%)",
    )
    fig.tight_layout()
    _save_plot_to_directory(fig, "trading_05_drawdowns")


def plot_regret_vs_number_of_experts_sensitivity(
    loss_matrix: np.ndarray,
) -> None:
    """
    Plot 6: Sensitivity — AdaHedge realised regret vs. number of experts N.

    For each N in {1, 2, 4, 8, 16, 36}, randomly samples N experts from
    the full set of 36 and runs AdaHedge on their loss submatrix.  Repeats
    for 15 random subsets (to average over sampling noise) and plots mean ± std.

    The theoretical O(sqrt(T ln N)) scaling (Theorem 3, phi=2) is overlaid as a
    reference.  Any deviation from the theoretical shape in the real data
    reflects correlations between experts not captured by the worst-case bound.
    """
    expert_subset_sizes_to_evaluate  = [1, 2, 4, 8, 16, 36]
    number_of_random_trials_per_size = 15
    
    print("\n  [SENSITIVITY] Regret vs. N experts ...")

    mean_final_regrets_per_n    = []
    std_final_regrets_per_n     = []
    theoretical_bounds_per_n    = []

    phi             = 2.0
    eulers_number   = np.exp(1.0)
    leading_constant_c_phi = phi * np.sqrt(phi ** 2 - 1.0) / (phi - 1.0)

    for number_of_experts_in_subset in expert_subset_sizes_to_evaluate:
        regrets_across_trials = []
        theory_bounds_across_trials = []

        for _ in range(number_of_random_trials_per_size):
            sampled_expert_indices = numpy_rng_phase2.choice(
                TOTAL_NUMBER_OF_EXPERTS,
                size=number_of_experts_in_subset,
                replace=False,
            )
            sampled_loss_submatrix = loss_matrix[:, sampled_expert_indices]

            if number_of_experts_in_subset == 1:
                # Single expert: regret is identically 0 (learner = best expert).
                # Skip AdaHedge to avoid degenerate eta underflow when log(K=1) = 0
                # makes the segment budget zero, causing eta to halve every round
                # until it underflows to 0.0 (producing inf × 0 = NaN warnings).
                regrets_across_trials.append(0.0)
                theory_bounds_across_trials.append(0.0)
            else:
                trial_adahedge_result = run_adahedge_algorithm(
                    loss_matrix=sampled_loss_submatrix,
                    segment_learning_rate_decay_factor=2.0,
                )
                regrets_across_trials.append(
                    float(trial_adahedge_result["cumulative_regret_of_learner"][-1])
                )

                # Theorem 3 bound using this trial's actual L*_T
                # (cumulative loss of the best expert in this subset).
                # Using the actual L*_T is mathematically correct — the bound
                # R_T <= c_phi * sqrt(4/(e-1) * L*_T * ln(K)) depends on the
                # realised best-expert loss, not a hardcoded approximation.
                trial_best_expert_cumulative_loss = float(
                    sampled_loss_submatrix.sum(axis=0).min()
                )
                trial_theory_bound = leading_constant_c_phi * np.sqrt(
                    (4.0 / (eulers_number - 1.0))
                    * trial_best_expert_cumulative_loss
                    * np.log(number_of_experts_in_subset)
                )
                theory_bounds_across_trials.append(trial_theory_bound)

        mean_regret       = float(np.mean(regrets_across_trials))
        std_regret        = float(np.std(regrets_across_trials))
        mean_theory_bound = float(np.mean(theory_bounds_across_trials))

        mean_final_regrets_per_n.append(mean_regret)
        std_final_regrets_per_n.append(std_regret)
        theoretical_bounds_per_n.append(mean_theory_bound)

        print(
            f"    N={number_of_experts_in_subset:2d}: "
            f"mean_regret={mean_regret:.2f} ± {std_regret:.2f}  "
            f"(theory~={mean_theory_bound:.2f})"
        )

    expert_count_array    = np.array(expert_subset_sizes_to_evaluate, dtype=float)
    mean_regrets_array    = np.array(mean_final_regrets_per_n)
    std_regrets_array     = np.array(std_final_regrets_per_n)
    theory_bounds_array   = np.array(theoretical_bounds_per_n)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        expert_count_array, theory_bounds_array,
        color=COLOR_THEORETICAL, linestyle="--", linewidth=1.8,
        label=r"Theorem 3 worst-case bound  $c_\phi\sqrt{\frac{4}{e-1}\,L_T^*\,\ln N}$",
        zorder=2,
    )
    ax.errorbar(
        expert_count_array, mean_regrets_array,
        yerr=std_regrets_array,
        color=COLOR_ADAHEDGE, linewidth=2.2,
        marker="o", markersize=7,
        capsize=4, capthick=1.5, elinewidth=1.2,
        label=(
            f"AdaHedge realised regret "
            f"(mean ± std, {number_of_random_trials_per_size} random subsets)"
        ),
        zorder=5,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(expert_subset_sizes_to_evaluate)
    ax.set_xticklabels([str(n) for n in expert_subset_sizes_to_evaluate])
    ax.legend(loc="upper left", fontsize=9, borderpad=0.6)
    _apply_dark_axis_styling(
        ax,
        title="Sensitivity: AdaHedge Regret vs. Number of Experts N",
        xlabel="Number of Experts N  (log₂ scale)",
        ylabel=r"Final Cumulative Regret $R_T$",
    )
    fig.tight_layout()
    _save_plot_to_directory(fig, "trading_06_regret_vs_n_experts")


def plot_transaction_cost_sweep_analysis(
    expert_signal_matrix: np.ndarray,
    expert_trading_return_matrix: np.ndarray,
    adahedge_weight_history: np.ndarray,
    return_day_date_index: pd.DatetimeIndex,
) -> None:
    """
    Plot 7: Cost sweep — AdaHedge wealth and Sharpe for c in {0, 5, 10, 25 bps}.

    Answers the core trading question: "Does AdaHedge's edge survive
    realistic transaction costs?"

    The weights are computed ONCE (from the loss matrix, which does not
    include costs) and are identical across all cost levels. Only the
    wealth path changes as the post-hoc cost deduction varies.

    Cost levels: {0, 5bp, 10bp, 25bp} where the base case (10bp) is
    highlighted in bold.
    """
    total_number_of_return_days = expert_trading_return_matrix.shape[0]

    # Compute gross portfolio returns once (same across all cost levels)
    daily_gross_portfolio_return_rates = np.array([
        float(np.dot(adahedge_weight_history[day_idx], expert_trading_return_matrix[day_idx]))
        for day_idx in range(total_number_of_return_days)
    ])

    cost_level_display_labels = ["c=0  (no cost)", "c=5bp", "c=10bp  (base)", "c=25bp"]
    cost_level_line_colors    = ["#06D6A0", "#FFD166", "#F72585", "#FF4D00"]

    print("\n  [COST SWEEP] c in {0, 5, 10, 25 bp} ...")

    fig, (ax_wealth, ax_sharpe) = plt.subplots(1, 2, figsize=(16, 5))

    for cost_coefficient, line_color, display_label in zip(
        TRANSACTION_COST_SWEEP_LEVELS, cost_level_line_colors, cost_level_display_labels
    ):
        daily_cost_fractions = compute_daily_transaction_costs_on_net_positions(
            expert_weight_history=adahedge_weight_history,
            expert_signal_matrix=expert_signal_matrix,
            transaction_cost_coefficient=cost_coefficient,
        )
        wealth_path    = _compound_daily_returns_to_wealth_path(
            daily_gross_portfolio_return_rates, daily_cost_fractions
        )
        rolling_sharpe = compute_rolling_annualised_sharpe_ratio(
            daily_gross_return_rates=daily_gross_portfolio_return_rates,
            daily_cost_fractions=daily_cost_fractions,
        )
        performance_metrics = compute_all_performance_metrics_for_strategy(
            cumulative_wealth_path=wealth_path,
            daily_gross_return_rates=daily_gross_portfolio_return_rates,
            daily_cost_fractions=daily_cost_fractions,
        )

        is_base_case = abs(cost_coefficient - TRANSACTION_COST_BASE_COEFFICIENT) < 1e-6
        line_width   = 2.5 if is_base_case else 1.4

        ann_ret_pct  = performance_metrics["annualised_cagr_fraction"] * 100
        sharpe_value = performance_metrics["sharpe_ratio_risk_free_zero"]

        ax_wealth.plot(
            return_day_date_index, wealth_path,
            color=line_color, linewidth=line_width,
            label=f"{display_label}  (ann.ret={ann_ret_pct:.1f}%)",
        )
        ax_sharpe.plot(
            return_day_date_index, rolling_sharpe,
            color=line_color, linewidth=line_width,
            label=f"{display_label}  (Sharpe={sharpe_value:.2f})",
        )

        print(
            f"    {display_label:<20} -> ann.ret={ann_ret_pct:.2f}%, "
            f"Sharpe={sharpe_value:.3f}, "
            f"max_DD={performance_metrics['maximum_drawdown_fraction']*100:.1f}%"
        )

    ax_wealth.axhline(
        y=INITIAL_PORTFOLIO_WEALTH_DOLLARS, color=_SPINE,
        linewidth=0.8, linestyle="--", alpha=0.5,
    )
    ax_wealth.legend(fontsize=8, loc="upper left")
    ax_wealth.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _apply_dark_axis_styling(
        ax_wealth,
        title="Cost Sweep — AdaHedge Cumulative Wealth",
        xlabel="Date",
        ylabel=f"Wealth (USD, start=${INITIAL_PORTFOLIO_WEALTH_DOLLARS:,.0f})",
    )

    ax_sharpe.axhline(y=0, color=_SPINE, linewidth=0.8, linestyle="--", alpha=0.5)
    ax_sharpe.legend(fontsize=8, loc="upper left")
    _apply_dark_axis_styling(
        ax_sharpe,
        title=f"Cost Sweep — Rolling {ROLLING_SHARPE_WINDOW_SIZE_IN_DAYS}d Sharpe",
        xlabel="Date",
        ylabel="Annualised Sharpe (Risk-Free=0)",
    )

    fig.suptitle(
        "AdaHedge Sensitivity to Transaction Costs  "
        "[c in {0, 5bp, 10bp, 25bp} per unit net turnover]",
        fontsize=12, fontweight="bold", color=_TITLE, y=1.015,
    )
    fig.tight_layout()
    _save_plot_to_directory(fig, "trading_07_cost_sweep")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — METRICS TABLE + REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save_performance_metrics_table(
    all_strategy_results: dict,
) -> None:
    """
    Print a formatted performance table to stdout and save to Reports/.

    Columns: CAGR%, Ann.Vol%, Sharpe (arith.mean), MaxDD%, Calmar, Ann.Cost%
    """
    separator_line = "=" * 118
    divider_line   = "-" * 118

    header_row = (
        f"{'Strategy':<38} {'CAGR%':>9} {'Ann.Vol%':>9} {'Sharpe*':>8} "
        f"{'MaxDD%':>8} {'Calmar':>8} {'Ann.Cost%':>10}"
    )

    print("\n" + separator_line)
    print("  PORTFOLIO PERFORMANCE METRICS SUMMARY")
    print(separator_line)
    print(header_row)
    print(divider_line)

    all_metrics_rows = []
    for strategy_name, result in all_strategy_results.items():
        metrics = compute_all_performance_metrics_for_strategy(
            cumulative_wealth_path=result["cumulative_wealth_path"],
            daily_gross_return_rates=result["daily_gross_portfolio_return_rates"],
            daily_cost_fractions=result["daily_transaction_cost_fractions"],
        )
        calmar_display = (
            f"{metrics['calmar_ratio']:.2f}" if not np.isnan(metrics["calmar_ratio"]) else "N/A"
        )
        row_text = (
            f"{strategy_name:<38} "
            f"{metrics['annualised_cagr_fraction']*100:>9.2f} "
            f"{metrics['annualised_volatility_fraction']*100:>9.2f} "
            f"{metrics['sharpe_ratio_risk_free_zero']:>8.3f} "
            f"{metrics['maximum_drawdown_fraction']*100:>8.2f} "
            f"{calmar_display:>8} "
            f"{metrics['annualised_cost_drag_fraction']*100:>10.3f}"
        )
        print(row_text)
        all_metrics_rows.append((strategy_name, metrics, row_text))

    print(separator_line)
    print(
        "  Note: Sharpe (*) uses arithmetic mean of daily returns x 252 in the numerator,\n"
        "        NOT CAGR -- consistent with the rolling Sharpe plots. Risk-free rate = 0.\n"
        "        All strategies shifted equally by rf=0; do not quote absolute Sharpe as\n"
        "        standalone 'fund quality' claims. CAGR% = compound annual growth rate.\n"
        f"       Backtest period: {all_metrics_rows[0][1]['number_of_years']:.1f} years  "
        f"       ({all_metrics_rows[0][1]['total_trading_days']} trading days)"
    )

    # Save to Reports/
    report_output_path = os.path.join(REPORTS_DIR, "trading_performance_metrics.txt")
    with open(report_output_path, "w", encoding="utf-8") as report_file:
        report_file.write("PORTFOLIO PERFORMANCE METRICS SUMMARY\n")
        report_file.write("Phase 2 — AdaHedge Algorithmic Trading Extension\n")
        report_file.write(separator_line + "\n")
        report_file.write(header_row + "\n")
        report_file.write(divider_line + "\n")
        for _, _, row_text in all_metrics_rows:
            report_file.write(row_text + "\n")
        report_file.write(separator_line + "\n")
        report_file.write(
            "Note: Sharpe (*) uses arithmetic mean of daily returns x 252 in the numerator,\n"
            "      NOT CAGR -- consistent with the rolling Sharpe plots. Risk-free rate = 0.\n"
            "      All strategies shifted equally by rf=0; do not quote absolute Sharpe as\n"
            "      standalone 'fund quality' claims. CAGR% = compound annual growth rate.\n"
        )

    print(f"\n  [SAVED] {report_output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Phase 2 main function.

    Execution sequence:
        1.  Run all 5 automated verification tests.
        2.  Download / load ETF price data.
        3.  Compute daily returns.
        4.  Build the 36-expert signal matrix.
        5.  Compute trading returns and the loss matrix.
        6.  Run all benchmark strategies.
        7.  Print and save the performance metrics table.
        8.  Generate all 7 plots.
        9.  Run sensitivity analyses (regret vs N, cost sweep).
        10. Stop
    """
    print("\n" + "=" * 72)
    print("  Phase 2 — Algorithmic Trading Extension")
    print(f"  Asset basket : {', '.join(SECTOR_ETF_TICKERS)}  ({NUMBER_OF_ASSETS_IN_BASKET} ETFs)")
    print(f"  Strategies   : {', '.join(ALL_STRATEGY_TYPES)}")
    print(f"  Experts      : {TOTAL_NUMBER_OF_EXPERTS}  (= {NUMBER_OF_STRATEGY_TYPES} × {NUMBER_OF_ASSETS_IN_BASKET})")
    print(f"  Initial wealth: ${INITIAL_PORTFOLIO_WEALTH_DOLLARS:,.0f}")
    print(f"  Cost base    : {TRANSACTION_COST_BASE_COEFFICIENT*10000:.0f} bp  "
          f"| sweep: {[int(c*10000) for c in TRANSACTION_COST_SWEEP_LEVELS]} bp")
    print("=" * 72)

    # ── Step 1: Automated tests ────────────────────────────────────────────────
    run_all_automated_verification_tests()

    # ── Step 2: Data acquisition ───────────────────────────────────────────────
    price_matrix, full_price_date_index = download_or_load_etf_adjusted_close_prices()
    total_number_of_price_rows   = price_matrix.shape[0]
    total_number_of_return_days  = total_number_of_price_rows - 1

    # Date index for return days (length T): date_index[1] corresponds to return[0]
    return_day_date_index = full_price_date_index[1:]

    print(f"\n  [INFO] Total return days: {total_number_of_return_days}")

    # ── Step 3: Daily asset returns ────────────────────────────────────────────
    asset_daily_return_matrix = compute_daily_simple_returns_from_prices(price_matrix)

    # ── Step 4: Signal matrix ──────────────────────────────────────────────────
    expert_signal_matrix = build_full_expert_signal_matrix(price_matrix)

    # ── Step 5: Trading returns and loss matrix ────────────────────────────────
    expert_trading_return_matrix = compute_expert_trading_return_matrix(
        expert_signal_matrix=expert_signal_matrix,
        asset_daily_return_matrix=asset_daily_return_matrix,
    )

    _, clipped_r_max = compute_loss_mapping_return_bound(
        expert_trading_return_matrix=expert_trading_return_matrix,
        clipping_percentile=RETURN_BOUND_CLIPPING_PERCENTILE,
    )

    loss_matrix = map_trading_returns_to_unit_interval_loss_matrix(
        expert_trading_return_matrix=expert_trading_return_matrix,
        maximum_return_bound=clipped_r_max,
    )
    print(
        f"\n  [LOSS] Matrix shape={loss_matrix.shape}, "
        f"range=[{loss_matrix.min():.4f}, {loss_matrix.max():.4f}], "
        f"mean={loss_matrix.mean():.4f}"
    )

    # ── Step 6: Compute worst-case optimal eta ───────────────────────────────────
    worst_case_optimal_learning_rate = np.sqrt(
        8.0 * np.log(TOTAL_NUMBER_OF_EXPERTS) / total_number_of_return_days
    )

    # ── Step 7: Run all strategies ─────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  RUNNING BACKTEST STRATEGIES")
    print("-" * 60)

    all_strategy_results = {}

    # AdaHedge (hero)
    print("  Running AdaHedge (phi=2) ...")
    adahedge_trading_result = run_adahedge_trading_strategy(
        expert_signal_matrix=expert_signal_matrix,
        expert_trading_return_matrix=expert_trading_return_matrix,
        loss_matrix=loss_matrix,
        transaction_cost_coefficient=TRANSACTION_COST_BASE_COEFFICIENT,
    )
    all_strategy_results["AdaHedge (phi=2)"] = adahedge_trading_result
    
    adahedge_hhi = compute_herfindahl_weight_concentration_index(
        adahedge_trading_result["expert_probability_allocation_weights"]
    )
    print(
        f"    -> Final regret   : {adahedge_trading_result['cumulative_regret_of_learner'][-1]:.4f}\n"
        f"    -> Segments used  : {adahedge_trading_result['total_number_of_segments_used']}\n"
        f"    -> Mean HHI concentration: {np.mean(adahedge_hhi):.4f} "
        f"(uniform CRP baseline = {1.0/TOTAL_NUMBER_OF_EXPERTS:.4f})"
    )

    # Fixed-eta Hedge variants
    for eta_label, eta_value in [
        ("Hedge eta=0.05",  0.05),
        ("Hedge eta=0.5",   0.50),
        ("Hedge eta=eta*(T)", worst_case_optimal_learning_rate),
    ]:
        print(f"  Running {eta_label}  (eta={eta_value:.5f}) ...")
        hedge_result = run_hedge_fixed_rate_trading_strategy(
            expert_signal_matrix=expert_signal_matrix,
            expert_trading_return_matrix=expert_trading_return_matrix,
            loss_matrix=loss_matrix,
            learning_rate_eta=eta_value,
            strategy_display_name=eta_label,
            transaction_cost_coefficient=TRANSACTION_COST_BASE_COEFFICIENT,
        )
        all_strategy_results[eta_label] = hedge_result
        
        hedge_hhi = compute_herfindahl_weight_concentration_index(
            hedge_result["expert_probability_allocation_weights"]
        )
        print(
            f"    -> Final regret   : {hedge_result['cumulative_regret_of_learner'][-1]:.4f}\n"
            f"    -> Mean HHI concentration: {np.mean(hedge_hhi):.4f}"
        )

    # Passive benchmarks
    print("  Running Buy-and-Hold EW ...")
    all_strategy_results["Buy-and-Hold EW"] = run_buy_and_hold_equal_weight_benchmark(
        asset_daily_return_matrix=asset_daily_return_matrix,
        transaction_cost_coefficient=TRANSACTION_COST_BASE_COEFFICIENT,
    )

    print("  Running Uniform CRP ...")
    all_strategy_results["Uniform CRP"] = run_uniform_constant_rebalanced_portfolio_benchmark(
        expert_signal_matrix=expert_signal_matrix,
        expert_trading_return_matrix=expert_trading_return_matrix,
        transaction_cost_coefficient=TRANSACTION_COST_BASE_COEFFICIENT,
    )

    print("  Running Best Expert in Hindsight ...")
    all_strategy_results["Best Expert (Hindsight)"] = run_best_expert_in_hindsight_benchmark(
        expert_signal_matrix=expert_signal_matrix,
        expert_trading_return_matrix=expert_trading_return_matrix,
        transaction_cost_coefficient=TRANSACTION_COST_BASE_COEFFICIENT,
    )

    # ── Step 8: Performance metrics table ─────────────────────────────────────
    print_and_save_performance_metrics_table(all_strategy_results)

    # ── Step 9: Generate plots ─────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  GENERATING PLOTS")
    print("-" * 60)

    plot_cumulative_wealth_comparison(
        all_strategy_results=all_strategy_results,
        return_day_date_index=return_day_date_index,
    )

    plot_weight_evolution_heatmaps(
        adahedge_weight_history=adahedge_trading_result["expert_probability_allocation_weights"],
        return_day_date_index=return_day_date_index,
    )

    plot_net_asset_positions_overlaid_on_prices(
        adahedge_weight_history=adahedge_trading_result["expert_probability_allocation_weights"],
        expert_signal_matrix=expert_signal_matrix,
        price_matrix=price_matrix,
        return_day_date_index=return_day_date_index,
    )

    plot_rolling_sharpe_ratio_comparison(
        all_strategy_results=all_strategy_results,
        return_day_date_index=return_day_date_index,
    )

    plot_drawdown_comparison(
        all_strategy_results=all_strategy_results,
        return_day_date_index=return_day_date_index,
    )

    # ── Step 10: Sensitivity analyses ─────────────────────────────────────────
    plot_regret_vs_number_of_experts_sensitivity(loss_matrix=loss_matrix)

    plot_transaction_cost_sweep_analysis(
        expert_signal_matrix=expert_signal_matrix,
        expert_trading_return_matrix=expert_trading_return_matrix,
        adahedge_weight_history=adahedge_trading_result["expert_probability_allocation_weights"],
        return_day_date_index=return_day_date_index,
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Phase 2 Backtest COMPLETE.")
    print(f"  Plots saved   -> {PLOTS_DIR}")
    print(f"  Metrics saved -> {REPORTS_DIR}")
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
