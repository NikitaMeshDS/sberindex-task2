import unittest
import numpy as np
import pandas as pd

try:
    from sberindex.forecasting.spatial_review import select_neighbors, neighbor_features
except ImportError:
    select_neighbors = neighbor_features = None


class SpatialReviewTests(unittest.TestCase):
    def test_single_orientation_is_symmetrized_and_ties_resolve_by_id(self):
        self.assertIsNotNone(select_neighbors)
        g = pd.DataFrame(
            {
                "territory_id_x": [3, 1, 3],
                "territory_id_y": [1, 2, 2],
                "distance": [0.0, 0.0, 5.0],
            }
        )
        n = select_neighbors(g, np.array([1, 2, 3]), 1)
        self.assertEqual(n[n.territory_id_x == 1].territory_id_y.tolist(), [2])
        self.assertEqual(n[n.territory_id_x == 2].territory_id_y.tolist(), [1])
        self.assertEqual(n[n.territory_id_x == 3].territory_id_y.tolist(), [1])

    def test_neighbor_future_changes_do_not_change_past_features(self):
        self.assertIsNotNone(neighbor_features)
        values = np.tile(np.arange(100.0, 124.0), (3, 1))
        neighbors = np.array([[1, 2], [0, 2], [0, 1]])
        a = neighbor_features(values, neighbors, 8, 0.15)
        values[:, 9:] = np.nan
        np.testing.assert_allclose(
            a, neighbor_features(values, neighbors, 8, 0.15), equal_nan=True
        )

    def test_previous_month_flags_and_missing_neighbors(self):
        self.assertIsNotNone(neighbor_features)
        values = np.array(
            [[100.0, 100.0, 100.0], [100.0, 200.0, 200.0], [100.0, np.nan, 100.0]]
        )
        neighbors = np.array([[1, 2], [0, 2], [0, 1]])
        x = neighbor_features(values, neighbors, 2, 0.15)
        np.testing.assert_allclose(x[0], [0.0, np.log(2), 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
