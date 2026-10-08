import itertools
import math
import unittest
import numpy as np
try:
    from sberindex.detection import bocpd_audit as audit
except ImportError:
    audit=None

class BocpdTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(audit,'BOCPD implementation missing')

    def test_matches_enumeration_of_all_partitions(self):
        x=np.array([.1,-.1,.8,.9]);hazard=1/12;sigma=.15
        score,posterior=audit.bocpd_scores(x[None,:],sigma,return_posterior=True)
        for n in range(2,len(x)+1):
            total=new=0.
            for cuts in itertools.product([0,1],repeat=n-1):
                starts=[0]+[j+1 for j,cut in enumerate(cuts) if cut]+[n]
                weight=hazard**sum(cuts)*(1-hazard)**(n-1-sum(cuts))
                for a,b in zip(starts[:-1],starts[1:]):
                    values=x[a:b];count=len(values);mean=values.mean();alpha=2+count/2
                    beta=sigma**2+.5*((values-mean)**2).sum()+.5*count/(1+count)*mean**2
                    logp=math.lgamma(alpha)-math.lgamma(2)+2*math.log(sigma**2)-alpha*math.log(beta)-.5*math.log(1+count)-count/2*math.log(2*math.pi)
                    weight*=math.exp(logp)
                total+=weight
                if cuts[-1]:new+=weight
            self.assertAlmostEqual(score[0,n-1],new/total,places=12)
        np.testing.assert_allclose(posterior.sum(axis=2),1.,atol=1e-12)

    def test_future_values_do_not_change_past_scores(self):
        a=np.array([[0.,.01,-.02,.01,.05,.4]])
        b=a.copy();b[:,4:]=50
        np.testing.assert_allclose(audit.bocpd_scores(a,.05)[:,:4],audit.bocpd_scores(b,.05)[:,:4],equal_nan=True)

    def test_gap_resets_and_first_point_has_no_score(self):
        x=np.array([[0.,0.,np.nan,1.,1.1]])
        s,p=audit.bocpd_scores(x,.05,return_posterior=True)
        self.assertTrue(np.isnan(s[0,[0,2,3]]).all())
        np.testing.assert_allclose(s[:,3:],audit.bocpd_scores(x[:,3:],.05),equal_nan=True)
        self.assertEqual(p[0,2,0],1.)

    def test_shift_probability_is_data_dependent(self):
        x=np.zeros((1,12));x[:,8:]=1.
        score=audit.bocpd_scores(x,.05)
        self.assertGreater(score[0,8],.8)
        self.assertLess(score[0,7],.1)

if __name__=='__main__':unittest.main()
