import numpy as np
import pandas as pd
from sberindex.detection.peer_residual_review import (
    regional_peer_residuals,
    detector_scores,
    alert_episodes,
    release_date,
)

SETTINGS = {
    "ewma_alpha": 0.45,
    "cusum_drift": 0.025,
    "noise_individual_weight": 0.5,
    "noise_floor": 0.0001,
    "threshold_quantile": 0.99,
    "bocpd_hazard": 1 / 12,
    "bocpd_mu": 0.0,
    "bocpd_kappa": 1.0,
    "bocpd_alpha": 2.0,
    "bocpd_sigma_floor": 0.02,
    "bocpd_mad_factor": 1.4826,
}
PARAMETERS = {
    "ewma": 0.07427838288053665,
    "spike": 0.14407071812623165,
    "cusum": 0.18829121580501101,
    "cusum_spike": 1.0506928196831102,
}


def fixture():
    return pd.DataFrame(
        dict(
            territory_id=[1, 2, 3, 4, 5, 6, 7],
            target=["2024-04"] * 7,
            region_code=[56] * 6 + [57],
            own_residual_log=[100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 999.0],
        )
    )


def test_excludes_focal_and_other_region():
    result = regional_peer_residuals(fixture())
    assert result.loc[0, "peer_count"] == 5
    assert result.loc[0, "peer_median_log"] == 3.0
    assert result.loc[0, "peer_residual_log"] == 97.0
    assert result.loc[1, "peer_median_log"] == 4.0


def test_min_peers_missing_focal_and_nan_not_zero():
    frame = fixture()
    frame.loc[1, "own_residual_log"] = np.nan
    result = regional_peer_residuals(frame)
    assert np.isnan(result.loc[0, "peer_residual_log"])
    assert result.loc[1, "peer_count"] == 5
    assert np.isnan(result.loc[1, "peer_residual_log"])
    assert np.isnan(result.loc[6, "peer_median_log"])


def test_future_invariance_peers_and_all_detectors():
    first = fixture()
    future = fixture()
    future["target"] = "2024-05"
    future.own_residual_log *= 10
    old = regional_peer_residuals(pd.concat([first, future], ignore_index=True))
    future.own_residual_log *= -100
    new = regional_peer_residuals(pd.concat([first, future], ignore_index=True))
    np.testing.assert_allclose(
        old.iloc[:7].peer_residual_log, new.iloc[:7].peer_residual_log, equal_nan=True
    )
    before = np.array([[0.02, -0.03, np.nan, 0.1, 0.2]])
    after = before.copy()
    after[:, 3:] = 999.0
    for method in PARAMETERS:
        np.testing.assert_allclose(
            detector_scores(before, method, PARAMETERS, SETTINGS)[:, :3],
            detector_scores(after, method, PARAMETERS, SETTINGS)[:, :3],
            equal_nan=True,
        )


def test_missing_month_resets_state_and_recovers():
    signal = np.array([[0.4, np.nan, 0.2]])
    assert np.isnan(detector_scores(signal, "ewma", PARAMETERS, SETTINGS)[0, 1])
    assert np.isclose(detector_scores(signal, "ewma", PARAMETERS, SETTINGS)[0, 2], 0.09)
    assert np.isclose(
        detector_scores(signal, "cusum", PARAMETERS, SETTINGS)[0, 2], 0.175
    )
    np.testing.assert_array_equal(
        alert_episodes([[True, False, True, True]]), [[True, False, True, False]]
    )


def test_release_calendar_lags():
    assert release_date("2024-04", 0) == pd.Timestamp("2024-05-01")
    assert release_date("2024-04", 2) == pd.Timestamp("2024-07-01")


def load_tests(loader, tests, pattern):
    """Register causal function tests with the project's unittest runner."""
    import unittest

    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )
