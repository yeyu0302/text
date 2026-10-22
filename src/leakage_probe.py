import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import folds_for
from src.features import FeatureBuilder
import optimize_pipeline as old

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    tr,te=pd.read_csv(root/'train.csv'),pd.read_csv(root/'test.csv')
    y=tr.label.to_numpy()-1;folds=folds_for(y)
    altered=y.copy();altered[folds==0]=(altered[folds==0]+1)%3
    a,_=old.build_clean_and_engineered_features(tr,te,y,folds)
    c,_=old.build_clean_and_engineered_features(tr,te,altered,folds)
    delta=np.abs(a.to_numpy()[folds!=0]-c.to_numpy()[folds!=0])
    changed=np.nan_to_num(delta)>1e-8
    fit=np.flatnonzero(folds!=0);val=np.flatnonzero(folds==0)
    x=tr.drop(columns='label')
    b,bv=FeatureBuilder('full').fit_transform(x.iloc[fit],y[fit],x.iloc[val])
    d,dv=FeatureBuilder('full').fit_transform(x.iloc[fit],altered[fit],x.iloc[val])
    pd.testing.assert_frame_equal(b,d);pd.testing.assert_frame_equal(bv,dv)
    report=dict(outer_fold=0,changed_validation_labels=int((folds==0).sum()),
                legacy_training_rows_affected=int(changed.any(axis=1).sum()),
                legacy_feature_columns_affected=int(changed.any(axis=0).sum()),
                legacy_maximum_absolute_change=float(np.nanmax(delta)),
                corrected_training_features_unchanged=True,corrected_validation_features_unchanged=True)
    (root/'results/leakage_probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
