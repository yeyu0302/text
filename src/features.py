"""Fold-local Gemini-derived features. No validation labels accepted by this API."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from optimize_pipeline import (REAL_CAT_COLS, GEO_COLS, DISC_COLS, REAL_BIN_COLS,
                               AGE_PROXIES, STOREY_PROXIES, AREA_PROXIES, HEIGHT_PROXIES)

PROXIES = AGE_PROXIES + STOREY_PROXIES + AREA_PROXIES + HEIGHT_PROXIES
SELECTED = REAL_CAT_COLS + GEO_COLS + DISC_COLS + REAL_BIN_COLS + PROXIES


def row_features(df, aggregate='mean'):
    """Deterministic row-local cleaning; raw features remain available separately.

    Trimmed CV aggregation removes one extreme per side for groups of >=4;
    the three-sensor area group retains its mean to avoid using one sensor only.
    """
    out = pd.DataFrame(index=df.index)
    def proxy(cols):
        a = df[cols].to_numpy(float)
        if aggregate == 'median':
            return np.nanmedian(a, axis=1)
        if aggregate == 'trimmed':
            a = np.sort(a, axis=1)
            if len(cols) >= 4:
                a = a[:, 1:-1]
        return np.nanmean(a, axis=1)
    age, storey, area, h = [proxy(x) for x in [AGE_PROXIES, STOREY_PROXIES, AREA_PROXIES, HEIGHT_PROXIES]]
    if aggregate == 'mean':
        h = .28*(df.f_186+df.f_151+df.f_20)+.16*df.f_80.clip(-40, 80)
    out['age_proxy_mean'], out['storey_proxy_mean'] = age, storey
    out['area_proxy_mean'], out['height_proxy_mean'] = area, h
    out['age_proxy_std'] = df[AGE_PROXIES].std(axis=1)
    out['is_age_999'] = (df.structure_age == 999).astype(float)
    out['is_age_nan'] = df.structure_age.isna().astype(float)
    out['clean_age'] = df.structure_age.mask(df.structure_age.isna() | (df.structure_age > 200),
                                            np.clip(np.round(age/10)*10, 0, 100))
    bad = (df.storey_count == 0) | (df.storey_count >= 99)
    out['clean_storey'] = df.storey_count.mask(bad, np.clip(np.round(storey), 1, 7))
    out.loc[bad & (df.f_209 == 'c2') & (storey < 2.2), 'clean_storey'] = 1
    bad = (df.footprint_area >= 1500) & (df.footprint_area % 500 == 0) & (abs(df.footprint_area/10-area) < abs(df.footprint_area-area))
    out['clean_area'] = df.footprint_area.mask(bad, df.footprint_area/10)
    height = df.f_165.mask((df.f_165 < 0) & (df.f_165 >= -48), -df.f_165)
    out['clean_height'] = height.mask(df.f_165.isna() | (df.f_165 < -48) | (df.f_165 >= 96), np.clip(np.round(np.asarray(h)/4)*4, 4, 64))
    out['age_blended'] = .8*out.clean_age+.2*np.clip(age, 0, 120)
    out['storey_blended'] = .8*out.clean_storey+.2*np.clip(storey, 1, 7)
    out['height_blended'] = .8*out.clean_height+.2*np.clip(h, 4, 64)
    out['log_clean_age'] = np.log1p(out.clean_age.clip(lower=0))
    out['log_clean_area'] = np.log1p(out.clean_area)
    out['f_119_clean'] = df.f_119.fillna(1)
    return out


def keys(df, cleaned):
    d = df.fillna('MISSING').astype(str)
    out = {g:d[g] for g in GEO_COLS}
    for name, cols in [('i141_182',['f_141','f_182']), ('i141_roof',['f_141','roofing_kind']),
                       ('i141_f62',['f_141','f_62']), ('i141_214',['f_141','f_214']),
                       ('i109_182_roof',['f_109','f_182','roofing_kind']),
                       ('i109_f62',['f_109','f_62']), ('i109_f69',['f_109','f_69']),
                       ('istruct',['roofing_kind','f_62','f_69','f_209','f_182','f_214'])]:
        out[name] = d[cols].agg('|'.join, axis=1)
    out['i109_storey'] = d.f_109+'|'+cleaned.clean_storey.fillna(-1).round().astype(str)
    out['i90_struct'] = d.f_90+'|'+out['istruct']
    out['i109_struct'] = d.f_109+'|'+out['i90_struct']
    return pd.DataFrame(out).reset_index(drop=True)


TE_SPECS = [('f_90',None,10,'f90'), ('f_109','f_90',7,'f109'),
            ('f_141','f_109',3,'f141_m3'), ('f_141','f_109',10,'f141_m10'),
            ('f_141','f_109',25,'f141_m25'), ('i141_182','f_141',4,'i141_182'),
            ('i141_roof','f_141',5,'i141_roof'), ('i141_f62','f_141',5,'i141_f62'),
            ('i141_214','f_141',4,'i141_214'), ('i109_182_roof','f_109',5,'i109_182_roof'),
            ('i109_f62','f_109',5,'i109_f62'), ('i109_f69','f_109',5,'i109_f69'),
            ('i109_storey','f_109',5,'i109_storey'), ('istruct',None,10,'istruct'),
            ('i90_struct','f_90',6,'i90_struct'), ('i109_struct','i90_struct',5,'i109_struct')]


def encode(fit_keys, y, apply_keys, smoothing=1., interactions=True):
    """All priors, parents and child statistics derive solely from y supplied here."""
    prior = np.bincount(y, minlength=3)/len(y)
    yoh = np.eye(3)[y]
    ans = {}
    for child, parent, m, prefix in TE_SPECS:
        if not interactions and child not in GEO_COLS:
            continue
        prior_app = np.tile(prior, (len(apply_keys), 1))
        if parent:
            par = pd.DataFrame(yoh).groupby(fit_keys[parent].to_numpy()).agg(['sum','count'])
            n = par[(0,'count')].to_numpy()[:,None]
            v = (np.column_stack([par[(k,'sum')] for k in range(3)])+8*smoothing*prior)/(n+8*smoothing)
            prior_app = pd.DataFrame(v, index=par.index).reindex(apply_keys[parent]).fillna(pd.Series(prior)).to_numpy()
        ch = pd.DataFrame(yoh).groupby(fit_keys[child].to_numpy()).agg(['sum','count'])
        n = ch[(0,'count')].reindex(apply_keys[child]).fillna(0).to_numpy()[:,None]
        sums = np.column_stack([ch[(k,'sum')].reindex(apply_keys[child]).fillna(0) for k in range(3)])
        p = (sums+m*smoothing*prior_app)/(n+m*smoothing)
        for k in range(3):
            ans[f'{prefix}_te_c{k}'] = p[:,k]
        ans[f'{prefix}_te_expected'] = p[:,1]+2*p[:,2]
    return pd.DataFrame(ans, dtype=np.float32)


class FeatureBuilder:
    def __init__(self, family='full', seed=42, smoothing=1., aggregate='mean', raw_mode='both', native=False):
        self.family, self.seed, self.smoothing = family, seed, smoothing
        self.aggregate, self.raw_mode, self.native = aggregate, raw_mode, native
        self.proxy = family in ('proxy','combined','structural','groups','full')
        self.te = family in ('hierarchy','combined','structural','groups','full')
        self.structural = family in ('structural','groups','full')
        self.groups = family in ('groups','full')

    def fit_base(self, x):
        self.ref = x.reset_index(drop=True).copy()
        self.clean_ref = row_features(self.ref, self.aggregate)
        self.raw_cols = list(x.drop(columns=['id','label'], errors='ignore').columns) if self.family == 'all_raw' else SELECTED
        self.cats = [c for c in self.raw_cols if x[c].dtype == object]
        if self.native:
            self.cats += [c for c in GEO_COLS if c not in self.cats]
        self.levels = {c:sorted(x[c].fillna('MISSING').astype(str).unique()) for c in self.cats}
        self.numeric = [c for c in self.raw_cols if c not in self.cats]
        self.group_maps = {}
        self.impute = {}
        for c, condition in [('f_103','f_113'),('f_175','f_104')]:
            self.impute[c] = (self.ref.groupby(['f_141',condition])[c].mean(),
                              self.ref.groupby(condition)[c].mean(), float(self.ref[c].mean()), condition)
        ref_stat = self._struct(self.ref, self.clean_ref.copy())
        for g in GEO_COLS:
            cols = ['clean_age','clean_storey','clean_area','clean_height','f_182','f_214','f_199','weak_masonry_score']
            group = ref_stat.groupby(self.ref[g])
            self.group_maps[g] = (self.ref[g].value_counts(), group[cols].mean(), ref_stat[cols].mean())
        return self

    def _struct(self, x, a):
        for c in REAL_BIN_COLS:
            a[c] = x[c]
        for c, (mapping, fallback, global_mean, condition) in self.impute.items():
            vals = mapping.reindex(pd.MultiIndex.from_frame(x[['f_141',condition]])).to_numpy()
            vals = pd.Series(vals, index=x.index).fillna(x[condition].map(fallback)).fillna(global_mean)
            a[c+'_imp'] = x[c].fillna(vals)
        a['height_per_storey'] = a.clean_height/a.clean_storey.clip(lower=1)
        a['slenderness'] = a.clean_height/np.sqrt(a.clean_area.clip(lower=50))
        a['area_per_storey'] = a.clean_area/a.clean_storey.clip(lower=1)
        a['volume_proxy'] = np.log1p(a.clean_area*a.clean_height)
        super_cols = ['f_182','f_214','f_199','f_103_imp','f_113','f_124','f_175_imp','f_23','f_104','f_226','f_1']
        a['superstructure_sum'] = a[super_cols].sum(axis=1)
        a['weak_masonry_score'] = 2*a.f_182+1.5*a.f_124+a.f_175_imp-2*a.f_214-2.5*a.f_199
        return a

    def base(self, x):
        x = x.reset_index(drop=True)
        out = x[self.numeric].astype(float).copy()
        if self.proxy and self.raw_mode == 'repaired':
            out = out.drop(columns=[c for c in DISC_COLS[:4]+PROXIES if c in out])
        for c in self.cats:
            s = x[c].fillna('MISSING').astype(str)
            if self.native:
                out[c] = pd.Categorical(s, categories=self.levels[c])
            else:
                for v in self.levels[c]:
                    out[c+'__'+v] = (s == v).astype(np.float32)
            freq = self.ref[c].fillna('MISSING').astype(str).value_counts(normalize=True)
            out[c+'_freq'] = s.map(freq).fillna(0).to_numpy()
        clean = row_features(x, self.aggregate)
        if self.proxy:
            out = pd.concat([out, clean], axis=1)
        if self.structural:
            struct = self._struct(x, clean.copy())
            out = pd.concat([out, struct[[c for c in struct if c not in out]]], axis=1)
        if self.groups:
            struct = self._struct(x, clean.copy())
            for g, (count, mapping, fallback) in self.group_maps.items():
                out[g+'_count'] = np.log1p(x[g].map(count).fillna(0))
                for c in mapping:
                    mean = x[g].map(mapping[c]).fillna(fallback[c])
                    out[g+'_mean_'+c] = mean
                    if c in ['clean_age','clean_height','clean_area','weak_masonry_score']:
                        out[g+'_diff_'+c] = struct[c]-mean
        # Tree libraries disallow some characters in feature names.
        out.columns = [f'x{i}' if self.native else c.replace('[','(').replace(']',')').replace('<','lt') for i,c in enumerate(out.columns)]
        return out

    def fit_transform(self, x, y, apply):
        x, apply = x.reset_index(drop=True), apply.reset_index(drop=True)
        self.fit_base(x)
        a, b = self.base(x), self.base(apply)
        if self.te:
            self.fit_keys = keys(x, self.clean_ref)
            self.y = np.asarray(y).copy()
            apply_keys = keys(apply, row_features(apply, self.aggregate))
            parts = []
            # Inner splits are independent of the outer holdout and its labels.
            for fit, val in StratifiedKFold(4, shuffle=True, random_state=self.seed+100).split(x,y):
                z = encode(self.fit_keys.iloc[fit].reset_index(drop=True), self.y[fit],
                           self.fit_keys.iloc[val].reset_index(drop=True), self.smoothing, self.family == 'full')
                z.index = val
                parts.append(z)
            ta = pd.concat(parts).sort_index()
            tb = encode(self.fit_keys,self.y,apply_keys,self.smoothing,self.family == 'full')
            a, b = pd.concat([a,ta],axis=1), pd.concat([b,tb],axis=1)
        # Stable names across train/validation; no global data-dependent feature schema.
        assert list(a.columns) == list(b.columns)
        return a, b
