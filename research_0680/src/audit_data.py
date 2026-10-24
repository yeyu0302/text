import json,hashlib,subprocess,sys,tempfile,os,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,PROJECT,RESULTS,read_data,dump,manifest,full_metrics

def run():
    x,y,test,sample=read_data();dest=RESULTS/'00_baseline';dest.mkdir(exist_ok=True)
    rows=[]
    for p in [ROOT/'results/best_submission.csv',ROOT/'results/alternative_submission.csv',ROOT/'results/legacy_reproduction/submission.csv',ROOT/'downloads/best_submission.csv']:
        if p.exists():
            s=pd.read_csv(p)
            rows.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(s),
                             valid=bool(list(s.columns)==['id','label'] and s.id.equals(sample.id) and s.label.isin([1,2,3]).all()),
                             kaggle_0679_mapping='unconfirmed; awaiting user submission identity'))
    pd.DataFrame(rows).to_csv(dest/'existing_submission_provenance.csv',index=False)
    expected=['SplineOrd_V4_Standalone.py','Single_SplineOrd_V4_NestedTE_Colab.ipynb','conditional_feature_selection_v7.py','methodology_report (2).md','Kaggle_0680_Improvement_Roadmap.md']
    found={name:[str(p) for p in ROOT.rglob(name) if '.git' not in p.parts] for name in expected}
    old=ROOT/'results/oof_predictions.csv'
    if old.exists():
        a=pd.read_csv(old);assert a.id.equals(x.id) and np.array_equal(a.label,y+1)
        p=a[[f'final_seed42_p{k}' for k in [1,2,3]]].to_numpy()
        historical=pd.DataFrame(dict(id=x.id,label=y+1,p1=p[:,0],p2=p[:,1],p3=p[:,2],pred=p.argmax(1)+1,fold=a.fold_seed42))
        historical.to_csv(dest/'historical_oof_aligned.csv',index=False)
        dump(dest/'historical_metrics.json',dict(**full_metrics(y,p),fold_definition='legacy class-wise shuffled round robin; NOT new sklearn folds',kaggle_mapping='not established'))
    provided=PROJECT/'results/00_baseline/provided/optimize_pipeline_user.py'
    if provided.exists():
        with tempfile.TemporaryDirectory(prefix='earthquake-s0-') as tmp:
            tmp=Path(tmp)
            for name in ['train.csv','test.csv']:(tmp/name).symlink_to(ROOT/name)
            result=subprocess.run([sys.executable,'-B',str(provided)],cwd=tmp,capture_output=True,text=True,timeout=300)
            (dest/'provided_reexecution.log').write_text(result.stdout+result.stderr)
            dump(dest/'provided_execution_status.json',dict(exit_code=result.returncode,source_sha256=hashlib.sha256(provided.read_bytes()).hexdigest(),
                faithful_unmodified_source=True,score_reproduced=False if result.returncode else 'inspect outputs',
                source_version='V6 header; V5 parameter metadata supplied separately'))
    dump(dest/'data_audit.json',manifest(dict(train_rows=len(x),test_rows=len(test),predictors=x.shape[1]-1,
        class_counts=np.bincount(y).tolist(),test_balance_evidence='provided competition PDF, preserved in repository',
        historical_files=found,kaggle_reported_by_user=.679,kaggle_submission_identity=None,
        validation_caveat='Historical feature definitions may retain earlier full-data exploration; all new fitted transformations and choices are outer-local.')))
    return x,y,test,sample

if __name__=='__main__':run()
