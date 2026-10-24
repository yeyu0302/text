import json
import numpy as np
import pandas as pd
from research_0680.src.common import read_data,dump,PROJECT
from src.features import row_features,AGE_PROXIES,AREA_PROXIES
from .paths import OUT

def run():
    x,y,test,_=read_data();clean=row_features(x);counts=x.f_141.value_counts();rows=[]
    for mode in ['raw','cleaned']:
        df=x.copy()
        if mode=='cleaned':df['structure_age']=clean.clean_age;df['storey_count']=clean.clean_storey
        for floor in [['f_69'],['f_209'],['f_69','f_209']]:
            cols=['f_141','structure_age','storey_count','f_182','f_214','roofing_kind','f_62']+floor
            signature=df[cols].fillna('MISSING').astype(str).agg('|'.join,axis=1)
            table=pd.crosstab(signature,y+1);conflict=(table>0).sum(1)>1
            rows.append(dict(mode=mode,floor_columns='+'.join(floor),columns=cols,conflicting_groups=int(conflict.sum()),
                 conflicting_rows=int(table.loc[conflict].sum().sum()),
                 empirical_majority_in_conflict=float(table.loc[conflict].max(axis=1).sum()/table.loc[conflict].sum().sum()),
                 in_sample_majority_all_rows=float(table.max(axis=1).sum()/len(x)),
                 interpretation='Subset-cell in-sample statistic, NOT Bayes ceiling for full observed X'))
    valid=x.structure_age.between(0,200)
    errors=x.loc[valid,AGE_PROXIES].subtract(x.loc[valid,'structure_age'],axis=0)
    mean=errors.mean(axis=1)
    flags=pd.DataFrame(dict(age=x.structure_age.isna()|(x.structure_age>=900)|(x.structure_age<0),
        storey=x.storey_count.isna()|(x.storey_count==0)|(x.storey_count>=90),
        area=(x.footprint_area>=1500)&(x.footprint_area%500==0)&(abs(x.footprint_area/10-x[AREA_PROXIES].mean(axis=1))<abs(x.footprint_area-x[AREA_PROXIES].mean(axis=1))),
        height=x.f_165.isna()|(x.f_165<0)|(x.f_165>=96)))
    out=OUT;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'partial_duplicate_definitions.csv',index=False)
    pd.DataFrame(dict(id=x.id,**{c:flags[c] for c in flags})).to_csv(out/'potential_measurement_anomaly_flags.csv',index=False)
    dump(out/'claim_audit.json',dict(predictor_count=x.shape[1]-1,train_rows=len(x),fine_groups=x.f_141.nunique(),
        test_fraction_train_group_count_le15=float((test.f_141.map(counts)<=15).mean()),
        full_raw_X_duplicate_rows=int(x.drop(columns='id').duplicated().sum()),
        anomaly_flag_any_row=float(flags.any(axis=1).mean()),anomaly_flag_cell_fraction=float(flags.to_numpy().mean()),
        anomaly_by_column=flags.mean().to_dict(),
        individual_age_proxy_residual_sd=errors.std().to_dict(),mean_age_proxy_mae=float(abs(mean).mean()),mean_age_proxy_residual_sd=float(mean.std()),
        disclaimers=['Anomaly flags do not establish deliberate contamination','MAE is not an irreducible error bound',
          'No universal Bayes ceiling follows from partial-feature duplicates','235 columns/175 pure-noise features is not an established fact; actual predictors=238']))
    print('Five-claim audit saved',flush=True)

if __name__=='__main__':run()
