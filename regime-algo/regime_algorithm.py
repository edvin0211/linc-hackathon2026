import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is importable when running from `regime-algo/`.
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading_simulator import TradingSimulator


# ===== LOAD DATA =====
prices = pd.read_csv(REPO_ROOT / "prices.csv", index_col="Date", parse_dates=True)
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Loaded {len(prices)} trading days, {len(prices.columns)} assets")


# ===== ENSEMBLE CONFIG (Selected Ranks 1–11 from `backtesting_cross_asset_regime_test_summary.csv`) =====
# (regime, defensive, cyclical)
triplets = [
    ("Idx_01", "FX_05", "Stock_11"),
    ("Idx_01", "FX_04", "Comm_06"),
    ("Idx_01", "FX_03", "Stock_02"),
    ("Idx_01", "FX_06", "Stock_07"),
    ("Idx_01", "Stock_08", "Comm_03"),
    ("Idx_01", "FX_02", "Stock_09"),
    ("Idx_01", "Stock_13", "Stock_10"),
    ("Idx_01", "Stock_05", "Stock_06"),
    ("Idx_01", "Comm_02", "Comm_05"),
    ("Idx_01", "FX_01", "Stock_03"),
    ("Idx_01", "Stock_01", "Stock_14"),
]

print(f"Using {len(triplets)} hardcoded triplets for ensemble.")


# ===== STRATEGY PARAMETERS (from backtesting_lab.ipynb) =====
REGIME_WINDOW = 40
VOL_WINDOW = 20
HIGH_STRESS_Z = 1.5
LOW_STRESS_Z = -0.5
TARGET_VOL = 0.012
MAX_GROSS_EXPOSURE = 0.95
INDEX_WEIGHT = 0.40
DEFENSIVE_WEIGHT = 1.00
CYCLICAL_WEIGHT = 1.00
INITIAL_CASH = 100_000
# Same default as `backtesting_lab.ipynb` (`SIGNAL_TO_RETURN_LAG` in the constants cell).
# That lag is applied inside the *vector* `compute_regime_triplet_backtest` path (shift before
# multiplying by returns). `TradingSimulator` separately fills orders at the *next* day's close,
# so simulator P&L will not exactly match the notebook's closed-form equity curve—only the same
# qualitative delay convention.
SIGNAL_TO_RETURN_LAG = 2


all_assets = sorted({a for t in triplets for a in t})
prices_assets = prices[all_assets].astype(float)


def compute_triplet_executed_weights(
    prices_df: pd.DataFrame,
    regime_asset: str,
    defensive_asset: str,
    cyclical_asset: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute executed weights and the per-triplet regime signal for one triplet.
    Logic matches `compute_regime_triplet_backtest` in backtesting_lab.ipynb.
    """
    asset_cols = [regime_asset, defensive_asset, cyclical_asset]
    asset_prices = prices_df[asset_cols].copy()
    asset_returns = asset_prices.pct_change().fillna(0.0)

    regime_mean = asset_prices[regime_asset].rolling(REGIME_WINDOW).mean()
    regime_std = asset_prices[regime_asset].rolling(REGIME_WINDOW).std()
    regime_z = (asset_prices[regime_asset] - regime_mean) / regime_std.replace(0, np.nan)

    base_weights = pd.Series(
        {
            regime_asset: -INDEX_WEIGHT,
            defensive_asset: DEFENSIVE_WEIGHT,
            cyclical_asset: -CYCLICAL_WEIGHT,
        },
        dtype=float,
    )
    base_weights = base_weights / base_weights.abs().sum()  # sum(|w|)=1

    basket_proxy_returns = (asset_returns * base_weights).sum(axis=1)
    basket_vol = basket_proxy_returns.rolling(
        VOL_WINDOW, min_periods=max(5, VOL_WINDOW // 2)
    ).std()

    signal = pd.Series(0.0, index=asset_prices.index)
    signal = signal.mask(regime_z > HIGH_STRESS_Z, 1.0)
    signal = signal.mask(regime_z < LOW_STRESS_Z, -1.0)

    gross_target = (TARGET_VOL / basket_vol).clip(upper=MAX_GROSS_EXPOSURE)
    gross_target = (
        gross_target.where(np.isfinite(gross_target) & (basket_vol > 0), 0.0)
        .fillna(0.0)
    )

    target_weights = pd.DataFrame(
        0.0, index=asset_prices.index, columns=asset_cols, dtype=float
    )
    for asset, base_weight in base_weights.items():
        target_weights[asset] = signal * gross_target * base_weight

    executed_weights = target_weights.shift(SIGNAL_TO_RETURN_LAG).fillna(0.0)
    return executed_weights, signal


def compute_target_shares_ensemble(prices_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build equal-weight ensemble of executed weights across selected triplets,
    then convert weights -> target shares.
    """
    executed_weights_ensemble = pd.DataFrame(
        0.0, index=prices_df.index, columns=all_assets, dtype=float
    )
    signal_ensemble = pd.Series(0.0, index=prices_df.index, dtype=float)

    for regime_asset, defensive_asset, cyclical_asset in triplets:
        executed_weights_triplet, signal_triplet = compute_triplet_executed_weights(
            prices_df=prices_df,
            regime_asset=regime_asset,
            defensive_asset=defensive_asset,
            cyclical_asset=cyclical_asset,
        )
        executed_weights_ensemble[executed_weights_triplet.columns] += executed_weights_triplet
        signal_ensemble += signal_triplet

    n = len(triplets)
    executed_weights_ensemble = executed_weights_ensemble / n
    signal_ensemble = signal_ensemble / n

    target_dollar = executed_weights_ensemble * INITIAL_CASH
    target_shares = (
        target_dollar.div(prices_df[all_assets].astype(float))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return target_shares, signal_ensemble


target_shares_df, signal_ensemble = compute_target_shares_ensemble(prices_assets)


def strategy(row_pos, cash, portfolio, signal_prices, data):
    date = data.index[row_pos]
    if date not in target_shares_df.index:
        return []

    targets = target_shares_df.loc[date]
    orders = []

    for ticker in all_assets:
        tgt = targets.get(ticker)
        if pd.isna(tgt):
            continue
        tgt_int = int(round(float(tgt)))
        delta = tgt_int - portfolio.get(ticker, 0)
        if delta == 0:
            continue
        action = "BUY" if delta > 0 else "SELL"
        orders.append((action, ticker, abs(delta)))

    return orders


# ===== RUN SIMULATION =====
simulator = TradingSimulator(assets=all_assets, initial_cash=INITIAL_CASH)
simulator.run(strategy, prices_assets, prices)
simulator.save_results(
    orders_file=str(OUTPUT_DIR / "macro_regime_orders.csv"),
    portfolio_file=str(OUTPUT_DIR / "macro_regime_portfolio.csv"),
)
simulator.plot_performance(prices_assets, save_file=str(OUTPUT_DIR / "macro_regime_performance_plot.png"))

# Stop here. The file currently contains leftover legacy code further down
# (from earlier iterations). Exiting early ensures we only generate the
# ensemble orders/portfolio artifacts once.
sys.exit(0)
