"""Audit submitted label changes and training-only soft-target scope risks."""
import os,itertools,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research_0680.src.common import read_data,dump,outer_folds
from src.features import row_features,REAL_BIN_COLS,REAL_CAT_COLS

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE6_RESULTS',str(HERE/'results')))

def group_probs(key,y):
    return pd.DataFrame(np.eye(3)[y]).groupby(np.asarray(key)).mean()

def run():
    OUT.mkdir(parents=True,exist_ok=True);train,y,test,sample=read_data()
    tables={i:pd.read_csv(HERE/f'evidence/submission_{i}.csv') for i in [3,4,5,6]}
    summary=[];changes=test[['id','f_90','f_109','f_141','structure_age','f_165']].copy()
    counts=train.f_141.value_counts();changes['train_village_n']=test.f_141.map(counts).fillna(0).astype(int)
    scores={3:.679,4:.677,5:.675,6:.672}
    for i,table in tables.items():
        assert table.id.equals(sample.id) and list(table.columns)==['id','label'] and table.label.isin([1,2,3]).all()
        changes[f'label_{i}']=table.label
        summary.append(dict(file=i,user_reported_kaggle_score=scores[i],rows=len(table),
            count1=int((table.label==1).sum()),count2=int((table.label==2).sum()),count3=int((table.label==3).sum()),
            changed_vs3=int((table.label!=tables[3].label).sum()),sha256=hashlib.sha256((HERE/f'evidence/submission_{i}.csv').read_bytes()).hexdigest()))
    pd.DataFrame(summary).to_csv(OUT/'submission_summary.csv',index=False)
    transition=[];subgroup=[]
    for old,new in [(3,4),(4,5),(5,6),(3,6)]:
        a=tables[old].label.to_numpy();b=tables[new].label.to_numpy();different=a!=b
        changes[f'changed_{old}_{new}']=different
        for i,j in itertools.product([1,2,3],repeat=2):transition.append(dict(old=old,new=new,from_label=i,to_label=j,n=int(np.sum((a==i)&(b==j)))))
        for label,mask in [('all',np.ones(len(a),bool)),('small_village_le15',changes.train_village_n<=15),('large_village_gt15',changes.train_village_n>15),
                           ('age_flag',test.structure_age.isna()|(test.structure_age>=900)|(test.structure_age<0)),
                           ('height_flag',test.f_165.isna()|(test.f_165<0)|(test.f_165>=96))]:
            subgroup.append(dict(old=old,new=new,subset=label,n=int(np.sum(mask)),changed=int(np.sum(mask&different)),change_rate=float(np.mean(different[mask])) if np.sum(mask) else None))
    changes.to_csv(OUT/'per_id_changes.csv',index=False);pd.DataFrame(transition).to_csv(OUT/'label_transitions.csv',index=False);pd.DataFrame(subgroup).to_csv(OUT/'change_subgroups.csv',index=False)
    cols=['f_141','structure_age','storey_count','f_182','f_214','roofing_kind','f_105','f_209']
    key=train[cols].fillna('MISSING').astype(str).agg('|'.join,axis=1);freq=pd.crosstab(key,y);conflict=(freq>0).sum(1)>1
    omitted=['footprint_area','f_165','f_119']+[c for c in REAL_BIN_COLS+REAL_CAT_COLS if c not in cols]
    variation=train[omitted].assign(signature=key).groupby('signature').nunique(dropna=False)
    dump(OUT/'partial_twins.json',dict(signature=cols,omitted=omitted,conflict_groups=int(conflict.sum()),
        conflict_rows=int(freq.loc[conflict].sum().sum()),conflict_groups_with_omitted_physical_variation=int((variation.loc[freq.index[conflict]].max(axis=1)>1).sum()),
        interpretation='Partial signature conflicts do not establish erroneous labels or a Bayes ceiling for the full X'))
    global_p=group_probs(key,y);folds=outer_folds(y,42);scope=[]
    for f in range(5):
        t=np.flatnonzero(folds!=f);v=np.flatnonzero(folds==f);local=group_probs(key.iloc[t],y[t])
        a=global_p.reindex(key.iloc[t]).to_numpy();b=local.reindex(key.iloc[t]).to_numpy()
        scope.append(dict(fold=f,train_rows=len(t),rows_affected_if_targets_were_precomputed_globally=int(np.sum(np.max(abs(a-b),axis=1)>1e-12)),
            shared_signatures=len(set(key.iloc[t])&set(key.iloc[v]))))
    pd.DataFrame(scope).to_csv(OUT/'hypothetical_soft_target_scope_risk.csv',index=False)
    rng=np.random.default_rng(12);assignments=list(itertools.permutations(range(3)))
    for _ in range(10000):
        p=rng.dirichlet(np.ones(3),size=3);sums=[sum(p[i,a[i]] for i in range(3)) for a in assignments];logs=[sum(np.log(p[i,a[i]]) for i in range(3)) for a in assignments]
        ia=int(np.argmax(sums));il=int(np.argmax(logs))
        if ia!=il:
            dump(OUT/'transport_objective_counterexample.json',dict(probabilities=p.tolist(),counts=[1,1,1],accuracy_assignment=assignments[ia],log_assignment=assignments[il],
                expected_correct_accuracy=sums[ia],expected_correct_log=sums[il],note='Mathematical counterexample only, no actual test labels or score estimate'));break
    dump(OUT/'audit_interpretation.json',dict(score_source='User reported',test_labels_available=False,
        net_level2_reduction_5_to6=146,level2_exits_5_to6=151,level2_entries_5_to6=5,total_changes_5_to6=160,
        warning='Net label-count reduction is not a count of corrected false positives. Soft-target scope table is a hypothetical risk demonstration, not a finding that the unavailable latest implementation leaked.'))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':run()
