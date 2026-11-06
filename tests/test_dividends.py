from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsx_div_ta.config import SignalConfig
from tsx_div_ta.dividends import dividend_features, infer_frequency, split_payments

CFG = SignalConfig()


def div_frame(
    n: int = 780,
    months: int = 3,
    amounts: list[float] | float = 1.0,
    price: float | np.ndarray = 100.0,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Business-day frame (default 2020-01-01..2022-12-28) with a payment every ``months``
    months starting 2020-01-11, snapped to the next trading day: a true calendar schedule.
    The default gives exactly 12 quarterly payments."""
    idx = pd.bdate_range(start, periods=n)
    div = np.zeros(n)
    pay_dates = pd.date_range(
        pd.Timestamp(start) + pd.DateOffset(days=10), idx[-1], freq=pd.DateOffset(months=months)
    )
    pay_idx = [int(idx.searchsorted(d)) for d in pay_dates if idx.searchsorted(d) < n]
    amts = amounts if isinstance(amounts, list) else [amounts] * len(pay_idx)
    for i, a in zip(pay_idx, amts, strict=False):
        div[i] = a
    raw = np.broadcast_to(np.asarray(price, dtype=float), (n,)).copy()
    return pd.DataFrame({"dividends": div, "raw_close": raw}, index=idx)


def test_fixture_has_twelve_quarterly_payments() -> None:
    assert int((div_frame()["dividends"] > 0).sum()) == 12


def test_constant_quarterly_ttm_yield() -> None:
    out = dividend_features(div_frame(), CFG)
    first_valid = out["ttm_yield"].first_valid_index()
    assert first_valid is not None
    assert out["ttm_yield"].loc[: out.index[150]].isna().all()  # fewer than 4 payments known
    assert (out["ttm_yield"].dropna().round(9) == 4.0).all()
    assert not out["div_cut"].any() and not out["div_suspended"].any()


def test_no_dividends() -> None:
    out = dividend_features(div_frame(amounts=0.0), CFG)
    assert (out["ttm_yield"].dropna() == 0).all()
    assert out["ttm_yield"].iloc[-1] == 0  # non-payer: 0 once a full year is covered
    assert not out["div_cut_recent"].any() and not out["div_suspended"].any()
    assert out["last_div"].isna().all()


def test_special_dividend_excluded() -> None:
    amounts = [1.0] * 6 + [5.0] + [1.0] * 6
    _reg, spec = split_payments(div_frame(amounts=amounts)["dividends"], 2.0)
    assert len(spec) == 1 and spec.iloc[0] == 5.0
    out = dividend_features(div_frame(amounts=amounts), CFG)
    assert out["special_div"].sum() == 5.0
    assert out["ttm_div"].max() <= 4.0 + 1e-9
    assert not out["div_cut"].any()  # payment after the special is compared to regular 1.0


def test_ttm_robust_to_pay_date_jitter() -> None:
    """A 365-day window would catch 5 payments when one date slips past a weekend; the
    last-N-payments definition must stay at exactly 4x."""
    f = div_frame(amounts=1.0)
    pays = f.index[f["dividends"] > 0]
    jittered = f.copy()
    jittered.loc[pays[4], "dividends"] = 0.0
    jittered.loc[pays[4] + pd.offsets.BDay(3), "dividends"] = 1.0  # one late payment
    out = dividend_features(jittered, CFG)
    assert (out["ttm_div"].dropna().round(9) == 4.0).all()


@pytest.mark.parametrize(
    ("gap", "freq"),
    [
        (91.0, 4.0),
        (30.0, 12.0),
        (182.0, 2.0),
        (365.0, 1.0),
        (70.0, 4.0),
        (0.0, None),
        (float("nan"), None),
    ],
)
def test_infer_frequency(gap: float, freq: float | None) -> None:
    got = infer_frequency(gap)
    assert np.isnan(got) if freq is None else got == freq


def test_empty_frame() -> None:
    out = dividend_features(div_frame(n=1).iloc[:0], CFG)
    assert out.empty


def test_first_payment_never_special() -> None:
    reg, spec = split_payments(pd.Series([0.0, 9.0, 0.0, 1.0]), 2.0)
    assert reg.tolist() == [9.0, 1.0] and spec.empty


@pytest.mark.parametrize(("new_amt", "is_cut"), [(0.85, True), (0.9, False), (0.95, False)])
def test_cut_threshold(new_amt: float, is_cut: bool) -> None:
    amounts = [1.0] * 8 + [new_amt] * 4
    out = dividend_features(div_frame(amounts=amounts), CFG)
    assert bool(out["div_cut"].any()) is is_cut
    assert bool(out["div_cut_recent"].iloc[-1]) is is_cut


def test_cut_recent_expires_after_lookback() -> None:
    amounts = [1.0] * 4 + [0.5] * 8  # cut ~2 years before the end
    out = dividend_features(div_frame(amounts=amounts), CFG)
    assert out["div_cut"].any()
    assert not out["div_cut_recent"].iloc[-1]


def test_suspension_detected() -> None:
    amounts = [1.0] * 6 + [0.0] * 6  # quarterly then stops
    out = dividend_features(div_frame(amounts=amounts), CFG)
    assert out["div_suspended"].iloc[-1]
    assert out["div_cut_recent"].iloc[-1]
    # not flagged while payments are still on schedule
    assert not out["div_suspended"].iloc[:380].any()


def test_monthly_payer_not_suspended() -> None:
    out = dividend_features(div_frame(months=1, amounts=0.1), CFG)
    assert not out["div_suspended"].any()
    assert (out["ttm_yield"].dropna().round(9) == 1.2).all()


def test_yield_percentile_rises_when_price_falls() -> None:
    price = np.r_[np.full(680, 100.0), np.full(100, 70.0)]
    out = dividend_features(div_frame(price=price), CFG)
    assert out["yield_pct"].iloc[-1] == pytest.approx(1.0)  # ties rank at the top (method="max")
    assert out["yield_pct"].iloc[670] == pytest.approx(1.0)  # flat yield history: also at the max
    assert out["yield_pct"].dropna().between(0, 1).all()


def test_dividend_growth() -> None:
    amounts = [1.0] * 8 + [1.1] * 4
    out = dividend_features(div_frame(amounts=amounts), CFG)
    assert out["div_growth"].iloc[-1] == pytest.approx(10.0, abs=0.01)


def test_features_are_trailing_only() -> None:
    f = div_frame(amounts=[1.0] * 6 + [0.5] * 7)
    full = dividend_features(f, CFG)
    part = dividend_features(f.iloc[:500], CFG)
    pd.testing.assert_frame_equal(part, full.iloc[:500])


def test_forward_yield_reflects_cut_immediately() -> None:
    amounts = [1.0] * 10 + [0.45] * 2  # ~55% cut, like a real telecom cut
    out = dividend_features(div_frame(amounts=amounts), CFG)
    assert out["fwd_yield"].iloc[-1] == pytest.approx(0.45 * 4)  # price 100
    assert out["ttm_yield"].iloc[-1] == pytest.approx(1.0 * 2 + 0.45 * 2)  # still has old pays
    assert out["div_cut_recent"].iloc[-1]


def test_forward_yield_zero_when_suspended_and_nan_for_non_payer() -> None:
    out = dividend_features(div_frame(amounts=[1.0] * 6 + [0.0] * 6), CFG)
    assert out["fwd_yield"].iloc[-1] == 0.0
    none = dividend_features(div_frame(amounts=0.0), CFG)
    assert none["fwd_yield"].isna().all()
