"""Exploratory associations only; never used for feature selection or CV fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from src.features import SELECTED

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    train=pd.read_csv(root/'train.csv');y=train.label.to_numpy()
    rows=[]
    for c in train.drop(columns=['id','label']):
        s=train[c];p=np.nan;effect=np.nan;stat=np.nan
        if s.dtype==object or s.nunique()<30:
            table=pd.crosstab(s.fillna('MISSING'),y)
            if len(table)>1:
                stat,p,_,_=chi2_contingency(table,correction=False)
                effect=np.sqrt(stat/(len(s)*min(table.shape[0]-1,2)))
            test='chi_square_cramers_v'
        else:
            values=[s[(y==k)&s.notna()] for k in [1,2,3]]
            if s.nunique()>1:
                stat,p=kruskal(*values)
                effect=max(0,(stat-2)/(s.notna().sum()-3))
            test='kruskal_epsilon_squared'
        rows.append(dict(feature=c,selected_hypothesis=c in SELECTED,test=test,p_value=p,effect=effect,
                         missing_rate=float(s.isna().mean()),n_unique=s.nunique()))
    df=pd.DataFrame(rows)
    valid=df.p_value.notna();order=df.loc[valid,'p_value'].sort_values().index
    q=df.loc[order,'p_value'].to_numpy()*len(order)/np.arange(1,len(order)+1)
    df.loc[order,'exploratory_fdr_q']=np.minimum(1,np.minimum.accumulate(q[::-1])[::-1])
    df.to_csv(root/'results/exploratory_feature_screen.csv',index=False)
    print(df.query('selected_hypothesis == False').sort_values('p_value').head(12).to_string(index=False))
