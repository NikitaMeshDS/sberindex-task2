import unittest
import numpy as np
import pandas as pd

try:
    import sberindex.detection.asof_detector_audit as audit
except ModuleNotFoundError:
    audit = None


class DetectorAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, 'as-of detector implementation is missing')

    def test_future_missingness_does_not_select_training_ids(self):
        panel = pd.DataFrame(np.ones((3,24)), index=[11,22,33])
        panel.iloc[1,19] = np.nan
        panel.iloc[2,15] = np.nan
        self.assertEqual(list(audit.select_cohort(panel).index), [11,22])
        changed = panel.copy();changed.iloc[:,18:] = np.nan
        self.assertEqual(list(audit.select_cohort(changed).index), [11,22])

    def test_missing_observation_resets_sequential_state(self):
        signal = np.array([[1.,np.nan,1.,1.]])
        np.testing.assert_allclose(audit.causal_scores(signal,'ewma'),[[.45,np.nan,.45,.6975]],equal_nan=True)
        np.testing.assert_allclose(audit.causal_scores(signal,'cusum'),[[.975,np.nan,.975,1.95]],equal_nan=True)
        np.testing.assert_allclose(audit.causal_scores(signal,'rolling_3m'),[[1.,np.nan,1.,1.]],equal_nan=True)

    def test_future_values_cannot_change_past_scores_or_noise_scale(self):
        values = np.arange(1,73,dtype=float).reshape(3,24)+100
        before = audit.centered_signal(values)
        after_values = values.copy();after_values[:,18:] *= np.array([[1.5],[.7],[2.]])
        after = audit.centered_signal(after_values)
        np.testing.assert_allclose(before[:,:6],after[:,:6])
        np.testing.assert_allclose(audit.noise_multiplier(before),audit.noise_multiplier(after))
        for method in audit.METHODS:
            np.testing.assert_allclose(audit.causal_scores(before,method)[:,:6],audit.causal_scores(after,method)[:,:6])

    def test_clean_scores_match_original_algorithms(self):
        from sberindex.detection.change_detection import score
        signal=np.random.default_rng(42).normal(size=(5,12))
        for method in audit.METHODS:
            np.testing.assert_allclose(audit.causal_scores(signal,method),score(signal,method))


if __name__ == '__main__':
    unittest.main()
