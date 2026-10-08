# Copyright 2026 Parth Maniar. Apache-2.0.
import numpy as np
import pandas as pd
KEY=["store_nbr","family"]
series=None
MODELS={'mean28':(28,False,False),'weekday56':(56,True,False),'weekday_log56':(56,True,True),'weekday_log112':(112,True,True),'promo_ridge112':(112,True,True),'promo_ridge56':(56,True,True)}
def forecast(hist,future,model):
    days,weekday,log=MODELS[model];cutoff=future.date.min()
    past=hist[(hist.date<cutoff)&(hist.date>=cutoff-pd.Timedelta(days=days))]
    assert len(past)>0 and past.date.max()<cutoff
    if model in ('promo_ridge112','promo_ridge56'):
        n=len(series);sid=past.series_id.to_numpy();dow=past.weekday.to_numpy()
        x=np.log1p(past.onpromotion.to_numpy(dtype=float));y=past.log_sales.to_numpy(dtype=float)
        xtx=np.zeros((n,8,8));xty=np.zeros((n,8))
        np.add.at(xtx,(sid,dow,dow),1)
        np.add.at(xtx,(sid,dow,7),x);np.add.at(xtx,(sid,7,dow),x)
        np.add.at(xtx,(sid,7,7),x*x)
        np.add.at(xty,(sid,dow),y);np.add.at(xty,(sid,7),x*y)
        penalties=np.array([.01]*7+[10.]);xtx+=np.diag(penalties)[None,:,:]
        beta=np.linalg.solve(xtx,xty[:,:,None])[:,:,0]
        fs=future.series_id.to_numpy();fd=future.weekday.to_numpy()
        pred_log=beta[fs,fd]+beta[fs,7]*np.log1p(future.onpromotion.to_numpy(dtype=float))
        p=np.expm1(np.maximum(pred_log,0))
        assert np.isfinite(beta).all() and np.isfinite(p).all() and (p>=0).all()
        return p
    keys=KEY+(['weekday'] if weekday else [])
    target='log_sales' if log else 'sales'
    mean=past.groupby(keys,observed=True)[target].mean().rename('prediction').reset_index()
    out=future[['id']+keys].merge(mean,on=keys,how='left',validate='many_to_one',sort=False)
    assert out.id.tolist()==future.id.tolist() and len(out)==len(future)
    assert out.prediction.notna().all(),'Missing series forecasts; no silent fallback'
    p=out.prediction.to_numpy(dtype=float)
    if log:p=np.expm1(p)
    assert np.isfinite(p).all() and (p>=0).all()
    return p
