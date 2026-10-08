"""Validated executable detector parameters, independent of forecast caches."""
import json
import math
from pathlib import Path
from sberindex.paths import ROOT

CONFIG_PATH = ROOT / 'configs/detectors.json'
SOURCES = ['configs/detectors.json', 'src/sberindex/detection/detector_settings.py']
KEYS = {'ewma_alpha', 'cusum_drift', 'noise_individual_weight', 'noise_floor',
        'threshold_quantile', 'bocpd_hazard', 'bocpd_mu', 'bocpd_kappa',
        'bocpd_alpha', 'bocpd_sigma_floor', 'bocpd_mad_factor'}


def validate_settings(settings):
    """Reject typo keys, booleans, nonfinite numbers and invalid model priors."""
    if not isinstance(settings, dict) or set(settings) != KEYS:
        raise ValueError('Detector settings must contain exactly the documented keys')
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
    for key in ['ewma_alpha', 'noise_individual_weight', 'threshold_quantile', 'bocpd_hazard']:
        if not 0 < settings[key] < 1:
            raise ValueError(f'{key} must lie strictly between 0 and 1')
    for key in ['noise_floor', 'bocpd_kappa', 'bocpd_sigma_floor', 'bocpd_mad_factor']:
        if settings[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if settings['cusum_drift'] < 0 or settings['bocpd_alpha'] <= 1:
        raise ValueError('CUSUM drift must be nonnegative and NIG alpha greater than 1')
    return dict(settings)


def load_settings(path=None):
    """Load the default JSON or an explicit path; values are never coerced."""
    return validate_settings(json.loads(Path(path or CONFIG_PATH).read_text()))
