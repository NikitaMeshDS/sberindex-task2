import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

class ResidualFeaturesTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'tools/residual_feature_experiment.py'
        spec=importlib.util.spec_from_file_location('experiment',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
        self.values=np.tile(np.arange(1.,25.),(2,6,1))
        self.profile=np.arange(1.,13.)
        self.events=pd.DataFrame({'available_from':['2023-02-11','2024-01-02'],'rate_after_pct':[7.5,20.],'delta_bps':[0,1250]})

    def test_features_exclude_future_and_keep_missing(self):
        original=self.m.feature_matrix(self.values,11,3,self.profile,self.events,'policy')
        changed=self.values.copy();changed[:,:,12:]=1e9
        np.testing.assert_array_equal(original,self.m.feature_matrix(changed,11,3,self.profile,self.events,'policy'))
        changed[0,1,11]=np.nan
        output=self.m.feature_matrix(changed,11,3,self.profile,self.events,'categories')
        self.assertTrue(np.isnan(output[0]).any())
        self.assertTrue(np.isfinite(output[1]).all())

    def test_future_policy_has_no_effect(self):
        a=self.m.feature_matrix(self.values,11,1,self.profile,self.events,'policy')
        altered=self.events.copy();altered.loc[1,'rate_after_pct']=9999
        np.testing.assert_array_equal(a,self.m.feature_matrix(self.values,11,1,self.profile,altered,'policy'))

    def test_fit_arrays_ignore_2024(self):
        a=self.m.training_arrays(self.values,self.profile,self.events,'categories')
        changed=self.values.copy();changed[:,:,12:]=np.nan
        b=self.m.training_arrays(changed,self.profile,self.events,'categories')
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_comparisons_preserve_saved_blend_name(self):
        rows=[dict(territory_id=1,origin='2024-06',target='2024-07',horizon=1,actual=10.,predicted=9.,model=m) for m in ['seasonal_pooled','prophet','blend_selected','residual_ridge_own']]
        result=self.m.summarize(pd.DataFrame(rows))
        self.assertEqual(set(result['paired'].reference),{'seasonal_pooled','prophet','blend_selected'})

    def test_macro_obeys_hypothetical_lag(self):
        macro=np.arange(100.,124.)
        a=self.m.feature_matrix(self.values,11,1,self.profile,self.events,'wage_snapshot',macro,2)
        changed=macro.copy();changed[10:]=1e9
        np.testing.assert_array_equal(a,self.m.feature_matrix(self.values,11,1,self.profile,self.events,'wage_snapshot',changed,2))
        self.assertAlmostEqual(a[0,-3],np.log(109.))

    def test_reference_includes_early_and_preserves_late(self):
        f=self.m.reference_frame()
        self.assertEqual(set(f[(f.horizon==1)&(f.origin<'2024-06')].target),{'2024-02','2024-03','2024-04','2024-05','2024-06'})
        source=pd.read_parquet(self.m.ROOT/'results/asof_cohort_scored.parquet')
        for model in ['prophet','seasonal_pooled','blend_selected']:
            a=source[source.model==model][self.m.KEYS+['model','actual','predicted']].sort_values(self.m.KEYS).reset_index(drop=True)
            b=f[(f.model==model)&((f.origin>='2024-06')|(f.horizon==12))].sort_values(self.m.KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b)

    def test_h12_has_only_available_saved_references(self):
        rows=[dict(territory_id=city,origin='2023-12',target='2024-12',horizon=12,actual=10.+city,predicted=9.,model=m) for city in [1,2] for m in ['seasonal_pooled','prophet','residual_ridge_own']]
        result=self.m.summarize(pd.DataFrame(rows))
        self.assertEqual(set(result['paired'].reference),{'seasonal_pooled','prophet'})

    def test_zero_correction_restores_seasonal(self):
        np.testing.assert_allclose(self.m.restore(self.values[:,0,11],11,3,self.profile,np.zeros(2),.5),self.values[:,0,11]*self.profile[2]/self.profile[11])
        np.testing.assert_allclose(self.m.restore(self.values[:,0,11],11,12,self.profile,np.array([100.,-100.]),.5),self.values[:,0,11]*np.exp([.5,-.5]))

if __name__=='__main__':unittest.main()
