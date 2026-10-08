import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sberindex.forecasting import operational_early_audit as audit
except ImportError:
    audit=None


class EarlyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit,'operational early implementation is missing')

    def test_windows_and_first_available_origin(self):
        self.assertEqual(audit.period_for('2024-02'),'early')
        self.assertEqual(audit.period_for('2024-09'),'late')
        self.assertEqual(audit.timing('2024-02',1),('2024-01','2023-12',2))
        with self.assertRaises(ValueError):audit.period_for('2024-07')

    def test_train_uses_only_2023_and_april_first_target(self):
        months=pd.period_range('2023-01',periods=24,freq='M').astype(str)
        data=pd.DataFrame(np.repeat(np.arange(1,25,dtype=float)[None,:],2,axis=0),columns=months)
        x,y,ids=audit.training_arrays(data)
        self.assertEqual(x.shape,(18,5))
        self.assertAlmostEqual(y[0],np.log(4/3))
        self.assertAlmostEqual(x[0,0],np.log(3))
        data.loc[:,data.columns>'2023-12']=np.nan
        changed_x,changed_y,changed_ids=audit.training_arrays(data)
        np.testing.assert_array_equal(x,changed_x)
        np.testing.assert_array_equal(y,changed_y)
        np.testing.assert_array_equal(ids,changed_ids)

    def test_recursion_does_not_replace_bridge_with_actual_january(self):
        class ConstantGrowth:
            def predict(self,x):return np.zeros(len(x))
        months=pd.period_range('2023-01',periods=24,freq='M')
        values=np.full((1,24),100.)
        values[:,12]=1e9
        prediction=audit.predict_asof(ConstantGrowth(),values,11,2,months,False)
        np.testing.assert_allclose(prediction,[[100.,100.]],rtol=1e-12)

    def test_refresh_rejects_omitted_provenance_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'results').mkdir()
            path=root/'results'/audit.CACHE[1]
            path.write_text(json.dumps({'input_sha256':{},'cached_artifact_sha256':{}}))
            with self.assertRaises(AssertionError):audit.refresh(root)
            hashes={}
            for relative in audit.INPUTS:
                source=root/relative;source.parent.mkdir(parents=True,exist_ok=True)
                source.write_bytes(b'fixture');hashes[relative]=audit.sha(source)
            path.write_text(json.dumps({'input_sha256':hashes,'cached_artifact_sha256':{}}))
            with self.assertRaises(AssertionError):audit.refresh(root)


if __name__=='__main__':unittest.main()
