"""Tests for tools/audit_ciks.py severity classification logic."""
from __future__ import annotations

from unittest.mock import patch


def _stub_latest_8k(behavior: dict[str, tuple[str | None, int]]):
    """Build a fake _latest_8k that returns from a {cik → (date_or_None, http_status)} map."""
    def fake(cik: str):
        return behavior.get(cik, (None, 200))
    return fake


def test_audit_classifies_broken_cik(monkeypatch):
    """A CIK returning non-200 is severity=broken; suggest the SEC current CIK."""
    from tools import audit_ciks

    monkeypatch.setattr(audit_ciks, "_load_sec_ticker_map", lambda: {"BAD": "0001999999"})
    monkeypatch.setattr(audit_ciks, "_latest_8k", _stub_latest_8k({
        "0001000000": (None, 404),     # ours: 404
        "0001999999": ("2026-04-15", 200),
    }))
    monkeypatch.setattr("ingestion.edgar_fundamentals_ingestion.CIK_MAP", {"BAD": "0001000000"})
    monkeypatch.setattr(audit_ciks.time, "sleep", lambda s: None)

    [r] = audit_ciks.audit(threshold_days=30, sleep_s=0)
    assert r.severity == "broken"
    assert r.suggested_cik == "0001999999"


def test_audit_classifies_stale_cik(monkeypatch):
    """A CIK that returns 200 but has stale 8-Ks vs SEC's current → severity=stale."""
    from tools import audit_ciks

    monkeypatch.setattr(audit_ciks, "_load_sec_ticker_map", lambda: {"OLD": "0001999999"})
    monkeypatch.setattr(audit_ciks, "_latest_8k", _stub_latest_8k({
        "0001000000": ("2020-01-15", 200),
        "0001999999": ("2026-04-15", 200),   # 6+ years fresher
    }))
    monkeypatch.setattr("ingestion.edgar_fundamentals_ingestion.CIK_MAP", {"OLD": "0001000000"})
    monkeypatch.setattr(audit_ciks.time, "sleep", lambda s: None)

    [r] = audit_ciks.audit(threshold_days=30, sleep_s=0)
    assert r.severity == "stale"
    assert r.suggested_cik == "0001999999"


def test_audit_classifies_minor_cik(monkeypatch):
    """SEC mismatch but ours is fresher or gap < threshold → severity=minor (no action)."""
    from tools import audit_ciks

    monkeypatch.setattr(audit_ciks, "_load_sec_ticker_map", lambda: {"FINE": "0001999999"})
    monkeypatch.setattr(audit_ciks, "_latest_8k", _stub_latest_8k({
        "0001000000": ("2026-04-22", 200),   # ours fresher
        "0001999999": ("2026-04-15", 200),
    }))
    monkeypatch.setattr("ingestion.edgar_fundamentals_ingestion.CIK_MAP", {"FINE": "0001000000"})
    monkeypatch.setattr(audit_ciks.time, "sleep", lambda s: None)

    [r] = audit_ciks.audit(threshold_days=30, sleep_s=0)
    assert r.severity == "minor"
    assert r.suggested_cik is None


def test_audit_classifies_ok_when_no_mismatch(monkeypatch):
    """SEC's current CIK matches ours → severity=ok."""
    from tools import audit_ciks

    monkeypatch.setattr(audit_ciks, "_load_sec_ticker_map", lambda: {"GOOD": "0001000000"})
    monkeypatch.setattr(audit_ciks, "_latest_8k", _stub_latest_8k({
        "0001000000": ("2026-04-22", 200),
    }))
    monkeypatch.setattr("ingestion.edgar_fundamentals_ingestion.CIK_MAP", {"GOOD": "0001000000"})
    monkeypatch.setattr(audit_ciks.time, "sleep", lambda s: None)

    [r] = audit_ciks.audit(threshold_days=30, sleep_s=0)
    assert r.severity == "ok"
    assert r.suggested_cik is None


def test_audit_threshold_governs_minor_vs_stale(monkeypatch):
    """Same gap classified differently based on threshold."""
    from tools import audit_ciks

    monkeypatch.setattr(audit_ciks, "_load_sec_ticker_map", lambda: {"X": "0001999999"})
    monkeypatch.setattr(audit_ciks, "_latest_8k", _stub_latest_8k({
        "0001000000": ("2026-04-01", 200),
        "0001999999": ("2026-04-21", 200),   # 20-day gap
    }))
    monkeypatch.setattr("ingestion.edgar_fundamentals_ingestion.CIK_MAP", {"X": "0001000000"})
    monkeypatch.setattr(audit_ciks.time, "sleep", lambda s: None)

    # threshold=30 → minor (gap 20 < 30)
    [minor] = audit_ciks.audit(threshold_days=30, sleep_s=0)
    assert minor.severity == "minor"

    # threshold=10 → stale (gap 20 > 10)
    [stale] = audit_ciks.audit(threshold_days=10, sleep_s=0)
    assert stale.severity == "stale"
