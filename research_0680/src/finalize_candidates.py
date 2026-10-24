"""Only predeclared nested rules passing both-seed gates can emit a candidate CSV."""
import json,time
import numpy as np
import pandas as pd
from .common import *
from .run_experiments import select_rule
from .train_baseline import predict_pipeline,FEATURE_CACHE

def validate(sub,sample):
    assert list(sub)==['id','label'] and sub.id.equals(sample.id) and sub.id.is_unique
    assert len(sub)==5100 and sub.label.isin([1,2,3]).all() and not sub.isna().any().any()

def run():
    rows=[];x,y,test,sample=read_data();dest=RESULTS/'05_final_candidates';dest.mkdir(exist_ok=True)
    for rule in ['proxy','adaptive']:
        a=dest/f'nested_{rule}_seed42/complete.json';b=dest/f'nested_{rule}_seed2026/complete.json'
        one=json.loads(a.read_text()) if a.exists() else {};two=json.loads(b.read_text()) if b.exists() else {}
        passed=bool(one and two and one.get('delta',0)>.001 and one.get('ci_low',-1)>0 and two.get('delta',0)>.001 and two.get('ci_low',-1)>0)
        reason='passed conservative local nested validation' if passed else 'unconfirmed: require delta > .001 and CI lower > 0 on BOTH seeds'
        record=dict(rule=rule,passed=passed,reason=reason,delta42=one.get('delta'),ci42_low=one.get('ci_low'),ci42_high=one.get('ci_high'),delta2026=two.get('delta'))
        if passed:
            start=time.perf_counter();FEATURE_CACHE.clear();choice,trace=select_rule(x,y,rule)
            p,factory=predict_pipeline(x,y,test,choice)
            # No quota constraint: no nested decision-rule validation justified it.
            sub=pd.DataFrame(dict(id=test.id,label=p.argmax(1)+1)).set_index('id').reindex(sample.id).reset_index()
            validate(sub,sample);name=f'submission_candidate_nested_{rule}.csv';path=dest/name
            if path.exists():raise FileExistsError('Preserve existing candidate')
            sub.to_csv(path,index=False)
            pd.DataFrame(dict(id=test.id,p1=p[:,0],p2=p[:,1],p3=p[:,2])).to_csv(dest/f'test_probabilities_nested_{rule}_full_refit.csv',index=False)
            dump(dest/f'candidate_nested_{rule}_manifest.json',manifest(dict(config=choice,inner_selection_trace=trace,
                 pipeline='Full retrain with cross-fitted train TE/KNN; full-fit test TE/KNN; no full-TE swap into fold models',
                 runtime_seconds=time.perf_counter()-start,counts=sub.label.value_counts().to_dict(),
                 kaggle_score=None,comparison_to_uploaded_0679_pipeline='unavailable because supplied generation code fails')))
            record['submission']=str(path)
        rows.append(record)
    pd.DataFrame(rows).to_csv(dest/'candidate_gate_decisions.csv',index=False)
    dump(dest/'candidate_policy.json',dict(maximum_candidates=2,actual_candidates=sum(r['passed'] for r in rows),
        no_unvalidated_submission_created=True,original_user_submission_preserved=True,quota_constraint_used=False,
        note='A passed local gate is not evidence that Kaggle Accuracy exceeds .680 or the supplied .679 score.'))
    print(rows,flush=True)

if __name__=='__main__':run()
