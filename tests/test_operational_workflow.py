import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'tools/operational_workflow.py'
        spec=importlib.util.spec_from_file_location('workflow',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_timing_and_training_gate(self):
        self.assertEqual(self.m.timing('2024-02',3,2),('2023-12','2024-05',5))
        self.assertFalse(self.m.training_available('2023-11','2023-12'))
        self.assertTrue(self.m.training_available('2023-12','2023-12'))

    def test_forecast_never_uses_unpublished_fact(self):
        p=np.ones(12);v=np.arange(1.,25.)
        def fake(history,h):return np.repeat(history[-1],h)
        a=self.m.forecast_row(v,p,'2024-12',1,3,.75,fake)
        v[23]=1e9
        b=self.m.forecast_row(v,p,'2024-12',1,3,.75,fake)
        self.assertEqual(a,b);self.assertEqual(a['effective_horizon'],4)
        self.assertEqual(a['predicted'],23.)

    def test_forecast_missing_history_refuses(self):
        v=np.ones(24);v[5]=np.nan
        r=self.m.forecast_row(v,np.ones(12),'2024-12',1,12,.75,lambda v,h:np.ones(h))
        self.assertEqual(r['status'],'missing_or_nonpositive_history')
        self.assertTrue(np.isnan(r['predicted']))

    def test_detector_selection_refuses_or_tiebreaks(self):
        table=pd.DataFrame({'method':['a','b','c'],'minimum_detection':[.7,.7,.8],'maximum_burden':[1.,.5,3.]})
        self.assertEqual(self.m.choose_detector(table,2.,.5),'b')
        self.assertIsNone(self.m.choose_detector(table,.1,.5))
        self.assertIsNone(self.m.choose_detector(table,2.,.9))

    def test_assignments_disjoint_required(self):
        with self.assertRaises(ValueError):self.m.validate_seeds([1,2],[2,3])
        self.m.validate_seeds([1,2],[3,4])

    def test_release_panel_rejects_duplicates_and_keeps_gaps(self):
        f=pd.DataFrame({'territory_id':[1,1], 'date':['2023-01','2023-03'], 'value':[1.,3.]})
        panel=self.m.release_panel(f,[1,2])
        self.assertTrue(np.isnan(panel.loc[1,'2023-02']))
        self.assertTrue(panel.loc[2].isna().all())
        with self.assertRaises(ValueError):self.m.release_panel(pd.concat([f,f]),[1])

if __name__=='__main__':unittest.main()
