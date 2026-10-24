import numpy as np
import pandas as pd
from .common import *
from .conditional_knn import ConditionalKNN

def run():
    x,y,_,_=read_data();dest=RESULTS/'01_data_forensics'
    folds=outer_folds(y,42);rows=[]
    for f in range(5):
        fit=np.flatnonzero(folds!=f);val=np.flatnonzero(folds==f)
        model=ConditionalKNN(10).fit(x.iloc[fit],y[fit]);model.transform(x.iloc[val])
        labels=dict(zip(x.id.iloc[fit],y[fit]+1))
        for i,idx in enumerate(val):
            neighbor=int(model.neighbor_ids[i,0]);rows.append(dict(id=int(x.id.iloc[idx]),true_label=int(y[idx]+1),
                neighbor_id=neighbor,neighbor_label=int(labels[neighbor]),nearest_distance=float(model.nearest_distances[i]),
                label_disagreement=bool(y[idx]+1!=labels[neighbor]),fold=f))
    near=pd.DataFrame(rows);near.to_csv(dest/'near_physical_neighbor_pairs.csv',index=False)
    near[near.label_disagreement].sort_values('nearest_distance').head(200).to_csv(dest/'near_conflicting_pairs_for_review.csv',index=False)
    suspect=pd.read_csv(dest/'suspicious_labels_all.csv')
    fields=['id','structure_age','storey_count','footprint_area','f_165','roofing_kind','f_90','f_109','f_141','f_182','f_214']
    result=suspect.merge(x[fields],on='id',validate='one_to_one').merge(near[['id','neighbor_id','neighbor_label','nearest_distance']],on='id',validate='one_to_one')
    result.head(200).to_csv(dest/'label_review_candidates.csv',index=False)
    print('Human-review tables only; labels unchanged',flush=True)

if __name__=='__main__':run()
