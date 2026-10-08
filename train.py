# Copyright 2026 Parth Maniar. Apache-2.0.
import json,time,hashlib,os
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
W=Path(os.environ.get('FORECAST_OUTPUT_DIR','outputs'));W.mkdir(parents=True,exist_ok=True);start=time.monotonic()
DATA=Path(os.environ.get('FORECAST_DATA_DIR','data'))
roots=[p.parent for p in DATA.rglob('train.csv') if (p.parent/'test.csv').exists() and (p.parent/'sample_submission.csv').exists()]
assert len(roots)==1,roots
root=roots[0]
train=pd.read_csv(root/'train.csv',parse_dates=['date'],dtype={'store_nbr':'int16','sales':'float32','onpromotion':'int32','family':'category'})
test=pd.read_csv(root/'test.csv',parse_dates=['date']);sample=pd.read_csv(root/'sample_submission.csv')
KEY=['store_nbr','family'];assert train.id.is_unique and test.id.is_unique and sample.id.tolist()==test.id.tolist()
assert not train.duplicated(['date']+KEY).any() and not test.duplicated(['date']+KEY).any()
assert train.sales.notna().all() and train.sales.ge(0).all()
assert train.onpromotion.notna().all() and test.onpromotion.notna().all()
assert train.onpromotion.ge(0).all() and test.onpromotion.ge(0).all()
series=pd.MultiIndex.from_frame(train[KEY]).unique()
train['series_id']=series.get_indexer(pd.MultiIndex.from_frame(train[KEY]))
test['series_id']=series.get_indexer(pd.MultiIndex.from_frame(test[KEY]))
assert train.series_id.ge(0).all() and test.series_id.ge(0).all()
assert test.date.min()==train.date.max()+pd.Timedelta(days=1)
horizon=(test.date.max()-test.date.min()).days+1
assert test.date.nunique()==horizon
train['weekday']=train.date.dt.dayofweek;test['weekday']=test.date.dt.dayofweek
train['log_sales']=np.log1p(train.sales)
print('Shapes',train.shape,test.shape,'horizon',horizon,'series',train.groupby(KEY,observed=True).ngroups)
print('Train dates',train.date.min(),train.date.max(),'test dates',test.date.min(),test.date.max())
print('Zero-sales rate',train.sales.eq(0).mean(),'all series retained')
train.groupby('date',observed=True).sales.sum().plot(figsize=(10,4),title='Daily total training sales');plt.ylabel('Sales');plt.tight_layout();plt.savefig(W/'daily_sales.png');plt.close()


# Copyright 2026 Parth Maniar. Apache-2.0.
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
rows=[];pred_rows=[];promo_rows=[]
for fold in [-3,-2,-1,0,1]:
    end=train.date.max()-pd.Timedelta(days=(1-fold)*horizon)
    cutoff=end-pd.Timedelta(days=horizon-1)
    va=train[(train.date>=cutoff)&(train.date<=end)].copy()
    assert va.date.nunique()==horizon and va.groupby(KEY,observed=True).ngroups==test.groupby(KEY).ngroups
    for model in MODELS:
        p=forecast(train,va,model);err=(np.log1p(p)-np.log1p(va.sales.to_numpy()))**2
        rows.append({'fold':fold,'model':model,'cutoff':str(cutoff.date()),'end':str(end.date()),'rows':len(va),'rmsle':float(np.sqrt(err.mean()))})
        per=va[['id','date']+KEY+['sales']].copy();per['fold']=fold;per['model']=model;per['prediction']=p;per['squared_log_error']=err;pred_rows.append(per)
        promo=va.onpromotion.to_numpy()>0
        for label,mask in [('zero_promotion',~promo),('positive_promotion',promo)]:
            if mask.any():promo_rows.append({'fold':fold,'model':model,'promotion_group':label,'rows':int(mask.sum()),'rmsle':float(np.sqrt(err[mask].mean()))})
        print(rows[-1])
metrics=pd.DataFrame(rows);metrics.to_csv(W/'fold_metrics.csv',index=False)
validation=pd.concat(pred_rows,ignore_index=True);validation.to_csv(W/'validation_predictions.csv',index=False)
summary=metrics.groupby('model').rmsle.agg(['mean','std']);summary.to_csv(W/'model_comparison.csv');print(summary)
best=summary['mean'].idxmin();print('Selected development model',best)
summary['mean'].plot.bar(figsize=(8,4),title='Rolling-origin RMSLE (lower is better)');plt.ylabel('RMSLE');plt.xticks(rotation=0);plt.tight_layout();plt.savefig(W/'model_comparison.png');plt.close()
per_family=validation.groupby(['model','family'],observed=True).squared_log_error.mean().pow(.5).rename('rmsle').reset_index();per_family.to_csv(W/'family_metrics.csv',index=False)

pd.DataFrame(promo_rows).to_csv(W/'promotion_group_metrics.csv',index=False)
paired=metrics.pivot(index='fold',columns='model',values='rmsle');paired['promo_minus_samewindow']=paired.promo_ridge112-paired.weekday_log112;paired['promo_minus_original']=paired.promo_ridge112-paired.weekday_log56;paired['promo56_minus_samewindow']=paired.promo_ridge56-paired.weekday_log56;paired['promo56_minus_promo112']=paired.promo_ridge56-paired.promo_ridge112;paired.to_csv(W/'paired_fold_deltas.csv');print(paired)


# Copyright 2026 Parth Maniar. Apache-2.0.
all_pred=test[['id','date']+KEY].copy()
for model in MODELS:all_pred[model]=forecast(train,test,model)
all_pred.to_csv(W/'test_predictions.csv',index=False)
sub=sample[['id']].merge(all_pred[['id',best]],on='id',how='left',validate='one_to_one',sort=False).rename(columns={best:'sales'})
assert sub.id.tolist()==sample.id.tolist() and len(sub)==len(test) and sub.sales.notna().all()
assert np.isfinite(sub.sales).all() and sub.sales.ge(0).all()
sub.to_csv(W/'submission.csv',index=False)
info={'selected_model':best,'validation_mean_rmsle':float(summary.loc[best,'mean']),'horizon_days':horizon,'train_rows':len(train),'test_rows':len(test),'folds':rows,'runtime_seconds':time.monotonic()-start,'candidate_public_score':None,'dropped_rows':0,'promotion_penalty':10,'weekday_penalty':0.01,'promotion_window_days':[56,112],'same_fold_baseline_control':True,'caveat':'Selection-biased development result. Promotion model ignores holidays and oil; fixed penalties and window, no hyperparameter search. Predicted log-sales floored at zero, targets untouched. No private leaderboard.'}
(W/'metrics.json').write_text(json.dumps(info,indent=2));print(info);print(sub.head())
manifest=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(W.iterdir()) if p.is_file()]
(W/'manifest.json').write_text(json.dumps(manifest,indent=2))
