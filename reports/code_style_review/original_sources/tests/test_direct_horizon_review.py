import unittest
import numpy as np
try:
 from sberindex.forecasting.direct_horizon_review import training_indices, feature_block
except ImportError:
 training_indices = feature_block = None

class DirectHorizonReviewTests(unittest.TestCase):
 def test_training_target_never_after_fit_origin_and_future_changes_irrelevant(self):
  self.assertIsNotNone(training_indices)
  values=np.full((4,24),100.)
  before=training_indices(values,11,3)
  values[:,12:]=np.nan
  after=training_indices(values,11,3)
  np.testing.assert_array_equal(before,after)
  self.assertTrue(all(target<=11 and target-source==3 for _,source,target in before))
  self.assertEqual(len(training_indices(values,11,12)),0)
 def test_missing_target_and_history_excluded_without_checking_future_intermediate(self):
  self.assertIsNotNone(training_indices)
  values=np.full((3,24),100.)
  values[0,5]=np.nan
  rows=training_indices(values,8,3)
  self.assertNotIn((0,2,5),map(tuple,rows))
  self.assertNotIn((0,5,8),map(tuple,rows))
  self.assertIn((0,3,6),map(tuple,rows))
 def test_all_covariates_and_aggregates_ignore_future(self):
  self.assertIsNotNone(feature_block)
  values=np.tile(np.arange(100.,124.),(4,1))
  cats=np.tile(values[:,None,:],(1,5,1));regions=np.array([1,1,2,2])
  x=feature_block(values,cats,regions,8,3,True)
  values[:,9:]=np.nan;cats[:,:,9:]*=900
  np.testing.assert_allclose(x,feature_block(values,cats,regions,8,3,True),equal_nan=True)
if __name__=='__main__':unittest.main()
