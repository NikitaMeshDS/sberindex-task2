import copy
import importlib
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

class DetectorSettingsTests(unittest.TestCase):
    def setUp(self):
        try:self.settings=importlib.import_module('sberindex.detection.detector_settings')
        except ModuleNotFoundError:self.fail('Executable detector settings module missing')

    def test_invalid_inputs_fail_early(self):
        good=json.loads(self.settings.CONFIG_PATH.read_text())
        for key,value in [('unknown',1),('ewma_alpha',0),('ewma_alpha',True),('bocpd_hazard',1),('bocpd_hazard',float('nan')),('noise_floor',0),('bocpd_alpha',1),('cusum_drift',-1)]:
            bad=copy.deepcopy(good);bad[key]=value
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'settings.json';p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError,msg=key):self.settings.load_settings(p)
        del good['bocpd_hazard']
        with self.assertRaises(ValueError):self.settings.validate_settings(good)

    def test_json_changes_reach_scores(self):
        from sberindex.detection.asof_detector_audit import causal_scores
        settings=self.settings.load_settings();settings.update(ewma_alpha=.25,cusum_drift=.1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'settings.json';p.write_text(json.dumps(settings))
            with patch.object(self.settings,'CONFIG_PATH',p):
                np.testing.assert_allclose(causal_scores(np.array([[1.,1.]]),'ewma'),[[.25,.4375]],rtol=0,atol=0)
                np.testing.assert_allclose(causal_scores(np.array([[.2,.2]]),'cusum'),[[.1,.2]],rtol=0,atol=1e-15)

    def test_noise_and_prior_settings_are_used(self):
        from sberindex.detection.asof_detector_audit import noise_multiplier
        from sberindex.detection.bocpd_audit import prior_sigma
        settings=self.settings.load_settings();settings.update(noise_individual_weight=.25,bocpd_sigma_floor=.7)
        z=np.array([[0,1,0,1,0,1],[0,2,0,2,0,2]],float)
        np.testing.assert_allclose(noise_multiplier(z,settings),[.75/(.25*.5+.75*.75),.75/(.25*1+.75*.75)])
        np.testing.assert_allclose(prior_sigma(np.zeros((2,6)),settings),.7)

    def test_nondefault_bocpd_matches_segment_marginals(self):
        from sberindex.detection.bocpd_audit import bocpd_scores
        settings=self.settings.load_settings();settings.update(bocpd_hazard=.2,bocpd_mu=.3,bocpd_kappa=2.,bocpd_alpha=3.)
        x=np.array([.1,.8]);sigma=.15;k=2.;a=3.;b=sigma**2;mu=.3
        def marginal(v):
            n=len(v);m=v.mean();beta=b+.5*((v-m)**2).sum()+.5*k*n/(k+n)*(m-mu)**2
            return math.exp(math.lgamma(a+n/2)-math.lgamma(a)+a*math.log(b)-(a+n/2)*math.log(beta)+.5*math.log(k/(k+n))-n/2*math.log(2*math.pi))
        reset=.2*marginal(x[:1])*marginal(x[1:]);grow=.8*marginal(x)
        scores,posterior=bocpd_scores(x[None,:],sigma,True,settings)
        self.assertAlmostEqual(scores[0,1],reset/(reset+grow),places=12)
        np.testing.assert_allclose(posterior.sum(axis=2),1.,atol=1e-12)

    def test_nondefault_prior_is_restored_after_gap(self):
        from sberindex.detection.bocpd_audit import bocpd_scores
        settings=self.settings.load_settings();settings.update(bocpd_mu=.3,bocpd_kappa=2.,bocpd_alpha=3.)
        x=np.array([[0.,.2,np.nan,.8,.9]])
        scores,posterior=bocpd_scores(x,.15,True,settings)
        np.testing.assert_allclose(scores[:,3:],bocpd_scores(x[:,3:],.15,settings=settings),equal_nan=True)
        self.assertEqual(posterior[0,2,0],1.)

if __name__=='__main__' :unittest.main()
