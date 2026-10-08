# Copyright 2026 Parth Maniar. Apache-2.0.
import unittest
import numpy as np
import pandas as pd
import forecast
class ForecastTests(unittest.TestCase):
    def setUp(self):
        forecast.series=pd.MultiIndex.from_tuples([(1,'A'),(2,'B')])
        dates=pd.date_range('2026-01-01',periods=160)
        self.history=pd.DataFrame([(i*2+k,d,k+1,'A' if k==0 else 'B',k,d.dayofweek,i%9,10.+k+i%7) for i,d in enumerate(dates) for k in (0,1)],columns=['id','date','store_nbr','family','series_id','weekday','onpromotion','sales'])
        self.history['log_sales']=np.log1p(self.history.sales)
        self.future=self.history.tail(32).copy()
    def test_all_models_are_finite_and_aligned(self):
        for name in forecast.MODELS:
            p=forecast.forecast(self.history,self.future,name)
            self.assertEqual(p.shape,(32,))
            self.assertTrue(np.isfinite(p).all() and (p>=0).all())
    def test_future_targets_cannot_change_predictions(self):
        changed=self.history.copy();changed.loc[changed.date>=self.future.date.min(),'sales']=99999
        changed.loc[changed.date>=self.future.date.min(),'log_sales']=np.log1p(99999)
        for name in forecast.MODELS:
            np.testing.assert_array_equal(forecast.forecast(self.history,self.future,name),forecast.forecast(changed,self.future,name))
if __name__=='__main__':unittest.main()
