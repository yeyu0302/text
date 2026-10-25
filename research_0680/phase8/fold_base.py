"""Freeze X-only category/region context from fit rows, including query schemas."""
import numpy as np
from .source_audit import provided

TARGETS=('clean_age','clean_storey','clean_area','clean_height','f_182','f_214','weak_masonry_score')
class FoldBase:
    def fit(self,x):
        self.source=provided();self.x=x.drop(columns='label',errors='ignore').reset_index(drop=True)
        self.base=self.source.clean_and_engineer_base_features(self.x,self.x.iloc[:0])
        self.numerics=[c for c in self.base.select_dtypes(include='number') if c not in self.source.GEO_HIERARCHY_COLS]
        self.freq={c:self.x[c].fillna('MISSING').astype(str).value_counts(normalize=True) for c in self.source.REAL_CATEGORICAL_COLS}
        self.groups={g:(self.base[g].value_counts(),self.base.groupby(g)[list(TARGETS)].mean()) for g in self.source.GEO_HIERARCHY_COLS}
        self.fallback=self.base[list(TARGETS)].mean()
        self.n_villages=self.base.groupby('f_109').f_141.nunique()
        return self
    def transform(self,q):
        out=self.source.clean_and_engineer_base_features(self.x,q.drop(columns='label',errors='ignore')).iloc[len(self.x):].reset_index(drop=True)
        for c,mapping in self.freq.items():out[c+'_freq']=out[c].map(mapping).fillna(0)
        for g,(counts,means) in self.groups.items():
            out[g+'_log_count']=np.log1p(out[g].map(counts).fillna(0))
            for c in TARGETS:
                v=out[g].map(means[c]).fillna(self.fallback[c]);out[g+'_mean_'+c]=v
                if g in ['f_141','f_109'] and c in ['clean_age','clean_height','clean_area','weak_masonry_score']:out[g+'_diff_'+c]=out[c]-v
        out['f109_n_villages']=out.f_109.map(self.n_villages).fillna(0)
        out['f141_share_of_f109']=np.expm1(out.f_141_log_count)/np.maximum(np.expm1(out.f_109_log_count),1)
        return out.reindex(columns=self.base.columns,fill_value=0)
