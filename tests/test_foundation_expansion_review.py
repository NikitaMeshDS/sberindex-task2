import unittest

import numpy as np
import pandas as pd

from sberindex.forecasting.foundation_expansion_review import (
    finetune_split,
    matched,
    ratio_history,
)


class FoundationExpansionReviewTests(unittest.TestCase):
    def test_ratio_history_causal_and_anchor_reconstructs(self):
        x = np.arange(1, 25, dtype=float)
        profile = np.arange(1, 13, dtype=float)
        history, scale = ratio_history(x, 13, profile)
        changed = x.copy()
        changed[14:] = 1e12
        other, _ = ratio_history(changed, 13, profile)
        np.testing.assert_array_equal(history, other)
        np.testing.assert_allclose(
            history * scale * profile[np.arange(14) % 12], x[:14]
        )

    def test_matching_rejects_incomplete_model_keys(self):
        keys = pd.DataFrame(
            {
                "territory_id": [1, 2],
                "origin": ["2023-12"] * 2,
                "target": ["2024-01"] * 2,
                "horizon": [1, 1],
            }
        )
        with self.assertRaises(ValueError):
            matched(keys.iloc[:1], keys)
        self.assertEqual(len(matched(keys, keys)), 2)

    def test_finetune_split_never_reads_2024(self):
        x = np.arange(48.0).reshape(2, 24)
        train, val = finetune_split(x)
        changed = x.copy()
        changed[:, 12:] = 1e10
        train2, val2 = finetune_split(changed)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(val, val2)
        self.assertEqual(train.shape, (2, 9))
        self.assertEqual(val.shape, (2, 12))

    def test_residual_batch_preparation_equals_each_series(self):
        profile = np.linspace(0.8, 1.2, 12)
        values = np.vstack([np.arange(1, 25), np.arange(1, 25) * 100])
        batch = np.stack([ratio_history(x, 15, profile)[0] for x in values])
        np.testing.assert_allclose(batch[0], batch[1], rtol=1e-6)
        np.testing.assert_array_equal(
            batch[1], ratio_history(values[1], 15, profile)[0]
        )


if __name__ == "__main__":
    unittest.main()
