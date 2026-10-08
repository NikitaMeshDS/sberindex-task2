import unittest
import numpy as np
try:
    from sberindex.detection import event_metric_review as audit
except ImportError:
    audit = None

class EventMetricReviewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, 'event benchmark module missing')

    def test_matching_is_one_to_one_and_maximum_cardinality(self):
        pairs = audit.match_events([5, 7], [6, 8], 1)
        self.assertEqual(pairs, [(5, 6), (7, 8)])
        self.assertEqual(audit.match_events([5], [4, 5, 6], 1), [(5, 5)])
        self.assertEqual(audit.match_events([5], [7], 1), [])

    def test_alarm_episodes_start_once_until_cleared(self):
        self.assertEqual(audit.alarm_events([False, True, True, False, True]), [1, 4])

    def test_forecast_never_reads_evaluation_values(self):
        x = np.tile(np.arange(12, dtype=float), 5)
        a = audit.forecast_residual(x, 36)
        x[40:] += 100
        b = audit.forecast_residual(x, 36)
        np.testing.assert_array_equal(a[:4], b[:4])
        np.testing.assert_allclose(b[4:] - a[4:], 100)

    def test_online_scores_do_not_use_future(self):
        x = np.random.default_rng(1).normal(0, .03, (3, 24))
        y = x.copy(); y[:, 12:] += .4
        for method in audit.ONLINE:
            np.testing.assert_allclose(audit.online_scores(x, method, .03)[:, :12],
                                       audit.online_scores(y, method, .03)[:, :12], equal_nan=True)

    def test_offline_boundary_is_first_changed_sample_and_drops_end(self):
        x = np.r_[np.zeros(12), np.full(12, .3)]
        for method in audit.OFFLINE:
            self.assertEqual(audit.offline_events(x, method, 2., .03), [12])

    def test_true_null_has_zero_labels(self):
        panel = audit.synthetic_panel(11, 5)
        self.assertEqual(len(panel), 15)
        for row in panel:
            self.assertEqual(len(row['truth']), {'no_change': 0, 'step': 1, 'pulse': 2}[row['shape']])

if __name__ == '__main__':
    unittest.main()
