import unittest
import numpy as np
import pandas as pd
try:
 from sberindex.forecasting.hierarchical_review import fit_profiles, predict_profile
except ImportError:
 fit_profiles = predict_profile = None

class HierarchicalReviewTests(unittest.TestCase):
 def test_fit_ignores_2024_values_and_missingness(self):
  self.assertIsNotNone(fit_profiles)
  a=np.tile(np.arange(1.,25.),(12,1)); a[:6,:12]*=2
  frame=pd.DataFrame(a,index=np.arange(12));regions=pd.Series([1]*6+[2]*6,index=frame.index)
  original=fit_profiles(frame,regions,2)
  frame.iloc[:,12:]=np.nan
  future_changed=fit_profiles(frame,regions,2)
  np.testing.assert_allclose(original[0],future_changed[0])
  for region in original[1]:np.testing.assert_allclose(original[1][region],future_changed[1][region])
 def test_small_region_falls_back_and_zero_weight_is_global(self):
  self.assertIsNotNone(predict_profile)
  glob=np.ones(12);regional={1:np.arange(1.,13.)/6.5}
  self.assertEqual(predict_profile(100.,0,3,9,glob,regional,.75),100.)
  self.assertEqual(predict_profile(100.,0,3,1,glob,regional,0.),100.)
if __name__=='__main__':unittest.main()
