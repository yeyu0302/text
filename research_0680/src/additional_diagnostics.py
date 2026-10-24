import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .common import *
from .error_attribution import encode_unsupervised

def run():
    x,y,test,sample=read_data();allx=pd.concat([x.drop(columns='id'),test.drop(columns='id')],ignore_index=True)
    source=np.r_[np.zeros(len(x)),np.ones(len(test))];results=[]
    for balanced in [False,True]:
        p=np.zeros(len(allx))
        for t,v in StratifiedKFold(3,shuffle=True,random_state=58).split(allx,source):
            a,b=encode_unsupervised(allx.iloc[t],allx.iloc[v]);w=np.ones(len(t));train=t<len(y)
            prior=np.bincount(y[t[train]],minlength=3)/train.sum()
            if balanced:w[train]=(1/3)/prior[y[t[train]]]
            clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,max_depth=3,l2_regularization=50,early_stopping=False,random_state=58).fit(a,source[t],sample_weight=w)
            p[v]=clf.predict_proba(b)[:,1]
        weight=np.ones(len(allx))
        if balanced:weight[:len(x)]=(1/3)/(np.bincount(y,minlength=3)/len(y))[y]
        results.append(dict(feature_set='all_238_no_id_no_label',prior_standardized=balanced,roc_auc=roc_auc_score(source,p,sample_weight=weight)))
    pd.DataFrame(results).to_csv(RESULTS/'01_data_forensics/adversarial_auc_all238.csv',index=False)
    base=pd.read_csv(RESULTS/'02_proxy_ablation/S2_proxy_V0/oof_predictions.csv').set_index('id')
    rows=[]
    for name,folder in [('residual','S5_residual'),('soft_weights','S5_soft_weights')]:
        q=pd.read_csv(RESULTS/'04_knn_ablation'/folder/'oof_predictions.csv').set_index('id').reindex(base.index)
        truth=(base.label.to_numpy()==2).astype(float)
        bmse=(truth-base.p2.to_numpy())**2;nmse=(truth-q.p2.to_numpy())**2
        rows.append(dict(method=name,mse_baseline=bmse.mean(),mse_modified=nmse.mean(),
                         balanced_mse_baseline=np.mean([bmse[base.label==k].mean() for k in [1,2,3]]),
                         balanced_mse_modified=np.mean([nmse[base.label==k].mean() for k in [1,2,3]]),
                         note='Independent development holdout, not final OOF. Damage labels unchanged.'))
    pd.DataFrame(rows).to_csv(RESULTS/'01_data_forensics/residual_and_weighting_holdout_mse.csv',index=False)
    print('Additional diagnostics',results,flush=True)

if __name__=='__main__':run()
