"""Phase 2.12 diagnostic baseline — TimesFM zero-shot vs our LGBM+RF+Ridge ensemble.

For each cell in {16 layers} × {65d, 252d}:
  - Use matured production prediction dates restricted to dates after the
    TimesFM 2.0 training-data cutoff (mid-2024) so TimesFM's input series
    are genuinely out-of-sample for it.
  - For each as_of date: TimesFM zero-shot forecasts every layer ticker's
    cumulative horizon log-return from its prior CONTEXT_LEN trading-day
    return history. The mean of the predicted log-returns over the horizon
    becomes TimesFM's per-ticker prediction.
  - Compute layer-restricted Spearman IC vs realized horizon returns.
  - Compare to the existing production ensemble's same-layer IC on the
    same dates (read from data/predictions/walkforward/.../predictions.parquet).

Verdict rules (pre-committed in the plan — no post-hoc tuning):
  - "TimesFM wins"  if mean Δ IC ≥ +0.010 AND TimesFM mean IC > 0
  - "Ensemble wins" if mean Δ IC ≤ -0.010
  - "Comparable"    otherwise

Outputs:
  - reports/timesfm/diagnostic_baseline.md  (markdown verdict table)
  - stdout summary

Run:
  python -m tools.timesfm_diagnostic_baseline

Cost: ~45-55 min CPU for the full 32-cell sweep. Subsequent runs much
faster if HuggingFace cached the model checkpoint. Disable GPU if you
have one — TimesFM's CPU path is fine for batch use.

This is a one-shot diagnostic to answer "should we add TimesFM as a
fourth ensemble member?" Re-run only if the model is retrained or the
holdout window changes materially.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ingestion.ticker_registry import tickers_in_layer, layers as all_layers
from tools.backtest_walk_forward import _load_realized

_PROJECT_ROOT = Path(__file__).parent.parent

# Per-horizon production cutoff + ablation per Phase 2.3A / 2.8 routing.
_CUTOFFS = {"65d": "2025-09-30", "252d": "2022-04-30"}
_ABLATIONS = {"65d": "no_ai_infra", "252d": "deep_only"}
_HORIZONS_TD = {"65d": 65, "252d": 252}

# TimesFM 2.0 training cutoff is ~mid-2024; restrict input sequences so
# the model can't have seen them.
_TFM_TRAINING_CUTOFF = date(2024, 6, 1)
_CONTEXT_LEN = 512
_MAX_DATES_252 = 250

_REPORT_DIR = _PROJECT_ROOT / "reports" / "timesfm"


def _trading_days_close(ticker: str, end_date: date, n_days: int) -> np.ndarray:
    """Last n_days of close prices for `ticker` strictly before `end_date`."""
    ohlcv_dir = _PROJECT_ROOT / "data" / "raw" / "financials" / "ohlcv" / ticker
    files = sorted(ohlcv_dir.glob("*.parquet"))
    if not files:
        return np.array([])
    df = (pl.concat([pl.read_parquet(f) for f in files])
            .select(["date", "close_price"])
            .filter(pl.col("date") < end_date)
            .sort("date")
            .tail(n_days))
    return df["close_price"].to_numpy().astype(np.float64)


def _load_timesfm(horizon_td: int):
    import timesfm
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=horizon_td,
            context_len=_CONTEXT_LEN,
            num_layers=50,   # 2.0 500M variant
            point_forecast_mode="median",
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch",
        ),
    )


def _spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ar = np.argsort(np.argsort(np.array(a)))
    br = np.argsort(np.argsort(np.array(b)))
    return float(np.corrcoef(ar, br)[0, 1])


def _matured_dates(horizon: str) -> list[date]:
    cutoff = _CUTOFFS[horizon]
    abl = _ABLATIONS[horizon]
    base = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}" / f"ablation={abl}")
    dates = sorted(date.fromisoformat(d.name.replace("date=", ""))
                   for d in base.glob("date=*"))
    out = [d for d in dates if d >= _TFM_TRAINING_CUTOFF]
    if horizon == "252d" and len(out) > _MAX_DATES_252:
        idx = np.linspace(0, len(out) - 1, _MAX_DATES_252).astype(int)
        out = [out[i] for i in idx]
    return out


def _ensemble_layer_ic(horizon: str, as_of: date, layer: str) -> float | None:
    cutoff = _CUTOFFS[horizon]
    abl = _ABLATIONS[horizon]
    horizon_td = _HORIZONS_TD[horizon]
    pred = (_PROJECT_ROOT / "data" / "predictions" / "walkforward"
            / f"cutoff={cutoff}" / f"ablation={abl}"
            / f"date={as_of.isoformat()}" / f"horizon={horizon}" / "predictions.parquet")
    if not pred.exists():
        return None
    df = pl.read_parquet(pred).filter(
        pl.col("layer") == layer,
        pl.col("expected_annual_return").is_not_null(),
    )
    if df.height < 3:
        return None
    realized = _load_realized(as_of, horizon_td, df["ticker"].to_list())
    aligned = [(p, realized[t])
               for t, p in zip(df["ticker"].to_list(), df["expected_annual_return"].to_list())
               if t in realized]
    if len(aligned) < 3:
        return None
    return _spearman([a[0] for a in aligned], [a[1] for a in aligned])


def _diagnose_horizon(horizon: str) -> list[dict]:
    horizon_td = _HORIZONS_TD[horizon]
    dates_list = _matured_dates(horizon)
    print(f"\n=== horizon={horizon} — {len(dates_list)} post-cutoff dates "
          f"({dates_list[0]} → {dates_list[-1]}) ===", flush=True)

    tfm = _load_timesfm(horizon_td)
    layers = list(all_layers())
    rows: list[dict] = []

    for i, as_of in enumerate(dates_list):
        contexts = []
        identities: list[tuple[str, str]] = []
        for layer in layers:
            for tk in tickers_in_layer(layer):
                hist = _trading_days_close(tk, as_of, _CONTEXT_LEN)
                if hist.size < 100:
                    continue
                lr = np.diff(np.log(hist)).astype(np.float32)
                contexts.append(lr)
                identities.append((layer, tk))
        if not contexts:
            continue
        t0 = time.time()
        try:
            point_fc, _ = tfm.forecast(contexts, freq=[0] * len(contexts))
        except Exception as e:  # noqa: BLE001 — fail-soft on a single date
            print(f"  {as_of}: forecast error {e}")
            continue
        cum_log_ret = point_fc.sum(axis=1)
        tfm_pred = {(lyr, tk): float(np.expm1(clr))
                    for (lyr, tk), clr in zip(identities, cum_log_ret)}
        elapsed = time.time() - t0

        scored_layers = 0
        for layer in layers:
            layer_tickers = tickers_in_layer(layer)
            tfm_pairs = [(tfm_pred[(layer, t)], t) for t in layer_tickers
                         if (layer, t) in tfm_pred]
            if len(tfm_pairs) < 3:
                continue
            realized = _load_realized(as_of, horizon_td, [t for _, t in tfm_pairs])
            aligned = [(p, realized[t]) for p, t in tfm_pairs if t in realized]
            if len(aligned) < 3:
                continue
            tfm_ic = _spearman([a[0] for a in aligned], [a[1] for a in aligned])
            ens_ic = _ensemble_layer_ic(horizon, as_of, layer)
            if ens_ic is None:
                continue
            rows.append({
                "horizon": horizon, "as_of": as_of, "layer": layer,
                "tfm_ic": tfm_ic, "ens_ic": ens_ic,
            })
            scored_layers += 1

        if i % 10 == 0 or i == len(dates_list) - 1:
            print(f"  [{i+1:>3d}/{len(dates_list)}]  {as_of}  "
                  f"{scored_layers} layers scored  ({elapsed:.1f}s)", flush=True)
    return rows


def _summarize(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(rows)
    if df.is_empty():
        return df
    return (df.group_by(["layer", "horizon"])
              .agg([
                  pl.col("tfm_ic").mean().alias("tfm_mean_ic"),
                  pl.col("ens_ic").mean().alias("ens_mean_ic"),
                  pl.col("tfm_ic").count().alias("n_dates"),
              ])
              .with_columns((pl.col("tfm_mean_ic") - pl.col("ens_mean_ic")).alias("delta_ic"))
              .with_columns(
                  pl.when((pl.col("delta_ic") >= 0.010) & (pl.col("tfm_mean_ic") > 0))
                    .then(pl.lit("TimesFM wins"))
                    .when(pl.col("delta_ic") <= -0.010)
                    .then(pl.lit("Ensemble wins"))
                    .otherwise(pl.lit("Comparable"))
                    .alias("verdict")
              )
              .sort(["horizon", "layer"]))


def _write_report(agg: pl.DataFrame) -> Path:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TimesFM diagnostic baseline (Phase 2.12)",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Restricted to post-TimesFM-2.0-training-cutoff dates (>= 2024-06-01) to "
        "avoid lookahead bias in TimesFM's input sequences.",
        "",
        "Verdict rules (pre-committed):",
        "- **TimesFM wins**: Δ IC ≥ +0.010 AND TimesFM mean IC > 0",
        "- **Ensemble wins**: Δ IC ≤ -0.010",
        "- **Comparable**: otherwise",
        "",
        "## Per-(layer, horizon) results",
        "",
        "| layer | horizon | n_dates | TimesFM IC | Ensemble IC | Δ IC | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in agg.iter_rows(named=True):
        lines.append(
            f"| {r['layer']} | {r['horizon']} | {r['n_dates']} | "
            f"{r['tfm_mean_ic']:+.4f} | {r['ens_mean_ic']:+.4f} | "
            f"{r['delta_ic']:+.4f} | {r['verdict']} |"
        )
    counts = agg.group_by(["verdict"]).agg(pl.len().alias("n")).sort("verdict")
    lines += ["", "## Verdict counts", "", "| verdict | n cells |", "|---|---:|"]
    for r in counts.iter_rows(named=True):
        lines.append(f"| {r['verdict']} | {r['n']} |")

    path = _REPORT_DIR / "diagnostic_baseline.md"
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    all_rows: list[dict] = []
    for h in ("65d", "252d"):
        all_rows.extend(_diagnose_horizon(h))
    if not all_rows:
        print("No rows scored.")
        return 1
    agg = _summarize(all_rows)
    print("\n" + "=" * 78)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(agg)
    print()
    counts = agg.group_by(["verdict"]).agg(pl.len().alias("n")).sort("n", descending=True)
    print(counts)
    out = _write_report(agg)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
