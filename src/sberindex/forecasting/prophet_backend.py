"""Historical Prophet configuration shared by the seven legacy pipelines.

Return raw predictions; each caller retains its original clipping policy.
No fit seed is added: this preserves the historical deterministic MAP calls.
Seasonal review variants have a separately frozen configuration and protocol.
"""

import numpy as np
import pandas as pd
from prophet import Prophet


def legacy_predict(history, training_dates, future_dates):
    history = np.asarray(history, dtype=float)
    training_dates = pd.DatetimeIndex(training_dates)
    future_dates = pd.DatetimeIndex(future_dates)
    if len(history) != len(training_dates) or len(history) < 2:
        raise ValueError("History and training dates must have matching lengths >=2")
    if not np.isfinite(history).all() or not len(future_dates):
        raise ValueError("Finite observed history and nonempty future dates required")
    if (
        not training_dates.is_monotonic_increasing
        or not future_dates.is_monotonic_increasing
    ):
        raise ValueError("Dates must be increasing")
    if (
        training_dates.has_duplicates
        or future_dates.has_duplicates
        or future_dates.min() <= training_dates.max()
    ):
        raise ValueError(
            "Future dates must be strictly after the unique training dates"
        )
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    model.fit(pd.DataFrame({"ds": training_dates, "y": history}))
    return model.predict(pd.DataFrame({"ds": future_dates})).yhat.to_numpy(float)
