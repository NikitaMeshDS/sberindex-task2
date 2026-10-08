import unittest
import numpy as np
try:
    from sberindex.detection import detector_delay_audit as audit
except ImportError:
    audit=None

class DetectorDelayTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(audit,'detector delay module missing')

    def test_alarm_arrives_later_without_backdating(self):
        alarm=np.zeros((2,6),bool);alarm[0,0]=True
        observed=np.ones_like(alarm);treated=np.array([True,False])
        m=audit.calendar_metrics(alarm,alarm,observed,treated,0,1)
        self.assertEqual(m['new_by_first_calendar_month'],0.)
        self.assertEqual(m['new_by_second_calendar_month'],1.)
        self.assertEqual(m['median_calendar_delay_if_new'],1.)
        self.assertEqual(audit.release_month('2024-07',1),'2024-08')
        self.assertEqual(audit.release_month('2024-06',2),'2024-08')

    def test_year_end_alarm_is_pending_not_detected(self):
        alarm=np.zeros((2,6),bool);alarm[0,5]=True
        observed=np.ones_like(alarm);treated=np.array([True,False])
        now=audit.calendar_metrics(alarm,alarm,observed,treated,0,0)
        late=audit.calendar_metrics(alarm,alarm,observed,treated,0,1)
        self.assertEqual(now['new_detected_by_december_n'],1)
        self.assertEqual(late['new_detected_by_december_n'],0)
        self.assertEqual(late['pending_observed_treated_months'],1)
        self.assertEqual(late['missing_treated_source_months'],0)

    def test_missing_and_common_three_month_denominator(self):
        alarm=np.zeros((3,6),bool);alarm[:2,0]=True
        observed=np.ones_like(alarm);observed[1,2]=False
        treated=np.array([True,True,False])
        for delay in [0,1,2]:
            m=audit.calendar_metrics(alarm,alarm,observed,treated,0,delay)
            self.assertEqual(m['treated_n'],2)
            self.assertEqual(m['evaluated_first_three_n'],1)
            self.assertEqual(m['incomplete_first_three_n'],1)
            self.assertEqual(m['new_by_third_calendar_month'],1.)
            self.assertEqual(m['missing_treated_source_months'],1)

if __name__=='__main__':unittest.main()
