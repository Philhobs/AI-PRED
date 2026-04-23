import datetime
import pytest
import polars as pl
from pathlib import Path


def _write_contracts(contracts_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    date_str = rows[0]["date"].isoformat()
    out_dir = contracts_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={
        "date": pl.Date, "awardee_name": pl.Utf8, "uei": pl.Utf8,
        "contract_value_usd": pl.Float64, "naics_code": pl.Utf8, "agency": pl.Utf8,
    }).write_parquet(out_dir / "awards.parquet")


def _write_ferc(ferc_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    date_str = rows[0]["snapshot_date"].isoformat()
    out_dir = ferc_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={
        "snapshot_date": pl.Date, "queue_date": pl.Date,
        "project_name": pl.Utf8, "mw": pl.Float64,
        "state": pl.Utf8, "fuel": pl.Utf8, "status": pl.Utf8, "iso": pl.Utf8,
    }).write_parquet(out_dir / "queue.parquet")


def _input_df(tickers: list[str], dates: list[datetime.date]) -> pl.DataFrame:
    rows = [{"ticker": t, "date": d} for t in tickers for d in dates]
    return pl.DataFrame(rows, schema={"ticker": pl.Utf8, "date": pl.Date})


def test_gov_contract_value_90d_correct(tmp_path):
    """gov_contract_value_90d sums contract_value_usd over rolling 90-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d1 = datetime.date(2024, 1, 1)
    d2 = datetime.date(2024, 1, 15)

    _write_contracts(contracts_dir, [
        {"date": d1, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
    ])
    _write_contracts(contracts_dir, [
        {"date": d2, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 500.0, "naics_code": "518210", "agency": "DOD"},
    ])

    result = join_gov_behavioral_features(_input_df(["NVDA"], [d2]), contracts_dir, ferc_dir)
    val = result.filter(pl.col("ticker") == "NVDA")["gov_contract_value_90d"][0]
    assert val == pytest.approx(1500.0)   # d1 (1000) + d2 (500)


def test_gov_contract_count_90d_correct(tmp_path):
    """gov_contract_count_90d counts individual award rows in rolling 90-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d = datetime.date(2024, 1, 15)

    _write_contracts(contracts_dir, [
        {"date": d, "awardee_name": "Microsoft Corporation", "uei": "U2",
         "contract_value_usd": 500.0, "naics_code": "541511", "agency": "GSA"},
        {"date": d, "awardee_name": "Microsoft Corporation", "uei": "U2",
         "contract_value_usd": 250.0, "naics_code": "541512", "agency": "GSA"},
    ])

    result = join_gov_behavioral_features(_input_df(["MSFT"], [d]), contracts_dir, ferc_dir)
    count = result.filter(pl.col("ticker") == "MSFT")["gov_contract_count_90d"][0]
    assert count == pytest.approx(2.0)


def test_gov_contract_momentum_positive_when_recent_exceeds_prior(tmp_path):
    """gov_contract_momentum > 0 when recent 30d awards exceed prior 60d awards."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    today = datetime.date(2024, 3, 1)
    old = datetime.date(2024, 1, 10)    # 51 days ago — in prior 60d window
    recent = datetime.date(2024, 2, 20) # 10 days ago — in 30d window

    _write_contracts(contracts_dir, [
        {"date": old, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 100.0, "naics_code": "518210", "agency": "DOD"},
    ])
    _write_contracts(contracts_dir, [
        {"date": recent, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
    ])

    result = join_gov_behavioral_features(_input_df(["NVDA"], [today]), contracts_dir, ferc_dir)
    momentum = result.filter(pl.col("ticker") == "NVDA")["gov_contract_momentum"][0]
    assert momentum > 0


def test_gov_ai_spend_30d_sums_all_awardees(tmp_path):
    """gov_ai_spend_30d sums only AI/DC NAICS contract values (not all vendors) in 30-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d = datetime.date(2024, 1, 15)

    _write_contracts(contracts_dir, [
        {"date": d, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
        {"date": d, "awardee_name": "Some Unmatched Company LLC", "uei": "U9",
         "contract_value_usd": 2000.0, "naics_code": "541511", "agency": "DHS"},
        {"date": d, "awardee_name": "Unrelated Vendor Inc", "uei": "U10",
         "contract_value_usd": 9999.0, "naics_code": "999999", "agency": "DOD"},
    ])

    result = join_gov_behavioral_features(_input_df(["NVDA"], [d]), contracts_dir, ferc_dir)
    spend = result["gov_ai_spend_30d"][0]
    assert spend == pytest.approx(3000.0)   # both awardees counted


def test_ferc_queue_mw_30d_sums_dc_state_mw(tmp_path):
    """ferc_queue_mw_30d sums MW for DC power states in rolling 30-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    snapshot = datetime.date(2024, 1, 1)
    queue_d = datetime.date(2024, 1, 10)   # within 30d of query_date Jan 20

    _write_ferc(ferc_dir, [
        {"snapshot_date": snapshot, "queue_date": queue_d, "project_name": "Solar VA",
         "mw": 300.0, "state": "VA", "fuel": "Solar", "status": "Active", "iso": "PJM"},
        {"snapshot_date": snapshot, "queue_date": queue_d, "project_name": "Wind TX",
         "mw": 200.0, "state": "TX", "fuel": "Wind", "status": "Active", "iso": "ERCOT"},
        {"snapshot_date": snapshot, "queue_date": queue_d, "project_name": "Offshore CA",
         "mw": 1000.0, "state": "CA", "fuel": "Wind", "status": "Active", "iso": "CAISO"},
    ])

    query_date = datetime.date(2024, 1, 20)
    result = join_gov_behavioral_features(_input_df(["NVDA"], [query_date]), contracts_dir, ferc_dir)
    mw = result["ferc_queue_mw_30d"][0]
    assert mw == pytest.approx(500.0)   # 300 + 200


def test_ferc_grid_constraint_score_above_one_when_spike(tmp_path):
    """ferc_grid_constraint_score > 1 when recent 30d MW well exceeds 12-month monthly average."""
    from processing.gov_behavioral_features import join_gov_behavioral_features
    import datetime as _dt

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    snapshot = _dt.date(2023, 1, 1)

    # 12 months of modest baseline: 10 MW/month
    baseline = [
        {"snapshot_date": snapshot,
         "queue_date": snapshot + _dt.timedelta(days=30 * i),
         "project_name": f"Base{i}", "mw": 10.0,
         "state": "VA", "fuel": "Solar", "status": "Active", "iso": "PJM"}
        for i in range(12)
    ]
    # Large spike in final month
    baseline.append({
        "snapshot_date": snapshot,
        "queue_date": _dt.date(2024, 1, 10),
        "project_name": "Spike", "mw": 500.0,
        "state": "VA", "fuel": "Solar", "status": "Active", "iso": "PJM",
    })
    _write_ferc(ferc_dir, baseline)

    query_date = _dt.date(2024, 1, 20)
    result = join_gov_behavioral_features(_input_df(["NVDA"], [query_date]), contracts_dir, ferc_dir)
    score = result["ferc_grid_constraint_score"][0]
    assert score > 1.0


def test_join_adds_exactly_six_columns(tmp_path):
    """join_gov_behavioral_features adds exactly 6 new columns."""
    from processing.gov_behavioral_features import join_gov_behavioral_features, GOV_BEHAVIORAL_FEATURE_COLS

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    df = _input_df(["NVDA"], [datetime.date(2024, 1, 15)])

    result = join_gov_behavioral_features(df, contracts_dir, ferc_dir)
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"
    assert len(result.columns) == len(df.columns) + 6


def test_ticker_with_no_contracts_zero_fills(tmp_path):
    """Ticker absent from contract awards gets 0.0 for all gov_contract_* columns."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d = datetime.date(2024, 1, 15)
    _write_contracts(contracts_dir, [
        {"date": d, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
    ])

    result = join_gov_behavioral_features(_input_df(["MSFT"], [d]), contracts_dir, ferc_dir)
    row = result.filter(pl.col("ticker") == "MSFT")
    assert row["gov_contract_value_90d"][0] == pytest.approx(0.0)
    assert row["gov_contract_count_90d"][0] == pytest.approx(0.0)
    assert row["gov_contract_momentum"][0] == pytest.approx(0.0)


def test_no_ferc_data_zero_fills(tmp_path):
    """Missing ferc_dir produces 0.0 for ferc_queue_mw_30d and ferc_grid_constraint_score."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"   # empty — no files

    result = join_gov_behavioral_features(
        _input_df(["NVDA"], [datetime.date(2024, 1, 15)]), contracts_dir, ferc_dir
    )
    assert result["ferc_queue_mw_30d"][0] == pytest.approx(0.0)
    assert result["ferc_grid_constraint_score"][0] == pytest.approx(0.0)


def test_no_contracts_dir_zero_fills_all_six(tmp_path):
    """Missing contracts_dir produces 0.0 for all 6 GOV_BEHAVIORAL_FEATURE_COLS."""
    from processing.gov_behavioral_features import join_gov_behavioral_features, GOV_BEHAVIORAL_FEATURE_COLS

    contracts_dir = tmp_path / "gov_contracts"   # does not exist
    ferc_dir = tmp_path / "ferc_queue"           # does not exist

    result = join_gov_behavioral_features(
        _input_df(["NVDA"], [datetime.date(2024, 1, 15)]), contracts_dir, ferc_dir
    )
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert result[col][0] == pytest.approx(0.0), f"{col} should be 0.0"
