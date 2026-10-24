"""Summarize frozen experiments; export a candidate only after the two-seed gate."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from research_0680.src.common import read_data, dump, full_metrics, outer_folds
from research_0680.src.build_proxy_features import GROUPS, trusted
from .methods import PosteriorProxy, predict
from .run import choose, clear
from .paths import OUT, REPORTS, REFERENCE_RESULTS


def reconstruction():
    x,y,_,_=read_data();rows=[]
    for fold,(t,v) in enumerate(KFold(5,shuffle=True,random_state=42).split(x)):
        builder=PosteriorProxy().fit(x.iloc[t]);query=x.iloc[v].reset_index(drop=True)
        values,_=builder.transform(query)
        for source,cols in GROUPS.items():
            name={'structure_age':'age','storey_count':'storey','footprint_area':'area','f_165':'height'}[source]
            good=trusted(query,source).to_numpy();truth=query[source].to_numpy(float)[good]
            estimates={'sensor_mean':query[cols].mean(axis=1).to_numpy()[good],
                       'posterior_mean':values[name+'_proxy_mean'].to_numpy()[good]}
            for method,estimate in estimates.items():
                error=estimate-truth;valid=np.isfinite(error);error=error[valid]
                rows.append(dict(fold=fold,source=source,method=method,n=len(error),
                                 mae=np.mean(abs(error)),mse=np.mean(error**2),
                                 note='Held-out trusted main is a silver reference, not latent ground truth; X only'))
    pd.DataFrame(rows).to_csv(OUT/'reconstruction_by_fold.csv',index=False)
    table=pd.DataFrame(rows);summary=[]
    for (source,method),part in table.groupby(['source','method']):
        n=part.n.sum();summary.append(dict(source=source,method=method,n=n,
            mae=np.average(part.mae,weights=part.n),rmse=np.sqrt(np.average(part.mse,weights=part.n))))
    pd.DataFrame(summary).to_csv(OUT/'reconstruction_summary.csv',index=False)


def finalize():
    REPORTS.mkdir(parents=True,exist_ok=True)
    comparison=pd.read_csv(OUT/'nested_comparison.csv');decisions=[]
    x,y,test,sample=read_data();subgroups=[];changes=[]
    for rule,part in comparison.groupby('rule'):
        passed=set(part.seed)=={42,2026} and bool(((part.delta>.001)&(part.ci_low>0)).all())
        decision=dict(rule=rule,passed=passed,gate='Both seeds: delta > 0.001 and ordinary paired bootstrap 95% CI lower > 0',
                      candidate=None)
        if passed:
            clear();config,trace=choose(x,y,rule);p=predict(x,y,test,config)
            path=OUT/f'submission_candidate_phase2_{rule}.csv'
            assert not path.exists(),'Preserve any existing result; use a fresh result directory'
            table=sample.copy();table['label']=p.argmax(1)+1;table.to_csv(path,index=False)
            dump(OUT/f'candidate_{rule}_selection.json',dict(config=config,trace=trace));decision['candidate']=str(path)
        decisions.append(decision)
        for seed in part.seed:
            dest=OUT/f'nested_{rule}_seed{seed}';oof=pd.read_csv(dest/'oof_predictions.csv')
            base=pd.read_csv(REFERENCE_RESULTS/'00_baseline'/f'fixed_baseline_seed{seed}'/'oof_predictions.csv')
            assert np.array_equal(oof.id,x.id) and np.array_equal(base.id,x.id)
            folds=outer_folds(y,int(seed));counts=np.zeros(len(x),int)
            for f in range(5):counts[folds==f]=x.loc[folds==f,'f_141'].map(x.loc[folds!=f,'f_141'].value_counts()).fillna(0).astype(int)
            for name,mask in [('train_group_le15',counts<=15),('train_group_gt15',counts>15),('all',np.ones(len(x),bool))]:
                for method,table in [('reference',base),(rule,oof)]:
                    met=full_metrics(y[mask],table.loc[mask,['p1','p2','p3']].to_numpy())
                    subgroups.append(dict(seed=seed,method=method,group=name,n=int(mask.sum()),**met))
            b=base.pred.to_numpy();a=oof.pred.to_numpy();l2=y==1
            changes.append(dict(seed=seed,rule=rule,level2_fixed=int(np.sum(l2&(b!=2)&(a==2))),
                level2_broken=int(np.sum(l2&(b==2)&(a!=2))),other_fixed=int(np.sum(~l2&(b!=y+1)&(a==y+1))),
                other_broken=int(np.sum(~l2&(b==y+1)&(a!=y+1)))))
    dump(OUT/'submission_gate.json',dict(decisions=decisions,known_kaggle_score='0.679: reported by user, no new Kaggle score measured',
        baseline_limitation='Uploaded .679-generating source fails on the current dataset; formal reference is a separately frozen reproducible model'))
    pd.DataFrame(subgroups).to_csv(OUT/'group_size_comparison.csv',index=False)
    pd.DataFrame(changes).to_csv(OUT/'level2_prediction_changes.csv',index=False)
    screen=pd.read_csv(OUT/'screening_summary.csv')
    log=pd.concat([screen.assign(scope='exploratory_dev_3750'),comparison.assign(scope='nested_oof_15000')],ignore_index=True)
    log.to_csv(OUT/'experiment_log.csv',index=False)
    print(json.dumps(decisions,ensure_ascii=False))


if __name__=='__main__':
    reconstruction();finalize()
