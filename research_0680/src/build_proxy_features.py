"""Silver-reference calibration; fitted using X only, never damage labels."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.features import row_features,AGE_PROXIES,STOREY_PROXIES,AREA_PROXIES,HEIGHT_PROXIES

GROUPS={'structure_age':AGE_PROXIES,'storey_count':STOREY_PROXIES,'footprint_area':AREA_PROXIES,'f_165':HEIGHT_PROXIES}
CLEAN={'structure_age':'clean_age','storey_count':'clean_storey','footprint_area':'clean_area','f_165':'clean_height'}

def trusted(x,source):
    s=x[source]
    bounds={'structure_age':(0,200),'storey_count':(1,7),'footprint_area':(50,1450),'f_165':(4,64)}
    lo,hi=bounds[source]
    return s.between(lo,hi)&s.notna()

def robust_line(x,y):
    a=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(a,y,rcond=None)[0]
    for _ in range(5):
        r=y-a@coef;scale=max(np.median(abs(r-np.median(r)))*1.4826,1e-5)
        w=np.minimum(1,1.35*scale/(abs(r)+1e-9));aw=a*np.sqrt(w[:,None])
        coef=np.linalg.lstsq(aw,y*np.sqrt(w),rcond=None)[0]
    return coef

class ProxyBuilder:
    def __init__(self,variant='V0'):self.variant=variant
    def fit(self,x):
        self.params={};self.calibration=[]
        for source,cols in GROUPS.items():
            valid=trusted(x,source);main=x.loc[valid,source].to_numpy(float)
            par=[]
            for c in cols:
                proxy=x.loc[valid,c].to_numpy(float);ok=np.isfinite(proxy)
                bias,slope=robust_line(main[ok],proxy[ok]);slope=float(slope)
                # A near-zero or negative slope cannot support inverse-sensor calibration.
                reliable=slope>.05
                slope=slope if reliable else 1.
                residual=(proxy[ok]-bias)/slope-main[ok]
                variance=max(float(np.mean(np.clip(residual,np.quantile(residual,.01),np.quantile(residual,.99))**2)),1e-5)
                par.append((float(bias),slope,variance,reliable))
                self.calibration.append(dict(source=source,proxy=c,bias=float(bias),slope=slope,residual_variance=variance,
                                             reliable=reliable,silver_n=int(ok.sum())))
            raw=x[cols].to_numpy(float)
            med=np.nanmedian(raw,axis=0);raw=np.where(np.isfinite(raw),raw,med)
            centered=raw-np.mean(raw,axis=0)
            pca=PCA(1,svd_solver='full').fit(centered)
            factors=pca.transform(centered).ravel()
            pcal=robust_line(factors[valid],main)
            self.params[source]=dict(params=np.asarray([p[:3] for p in par]),median=med,
                                     reliable=np.array([p[3] for p in par]),pca=pca,center=raw.mean(0),pcal=pcal,
                                     main_median=float(np.median(main)))
        return self
    def transform(self,x):
        base=row_features(x).copy();uncert={};repairs=[]
        for source,cols in GROUPS.items():
            par=self.params[source];a=x[cols].to_numpy(float)
            filled=np.where(np.isfinite(a),a,par['median'])
            bias,slope,var=par['params'].T
            z=(filled-bias)/slope
            weights=par['reliable']/var
            if weights.sum()==0:weights=np.ones(len(cols))
            weights=weights/weights.sum();latent=z@weights
            if self.variant=='V4':
                f=par['pca'].transform(filled-par['center']).ravel();latent=par['pcal'][0]+par['pcal'][1]*f
            spread=np.std(z,axis=1)
            m=x[source].to_numpy(float);valid=trusted(x,source).to_numpy()
            uncert.update({f'{source}_proxy_std':spread,
                           f'{source}_proxy_mad':np.median(abs(z-np.median(z,axis=1)[:,None]),axis=1),
                           f'{source}_proxy_iqr':np.quantile(z,.75,axis=1)-np.quantile(z,.25,axis=1),
                           f'{source}_proxy_range':np.ptp(z,axis=1),f'{source}_proxy_missing':np.isnan(a).sum(1),
                           f'{source}_main_gap':np.where(np.isfinite(m),m-latent,0),
                           f'{source}_trust_flag':valid.astype(float),
                           f'{source}_uncertainty':np.sqrt(1/np.maximum(np.sum(par['reliable']/var),1e-6))*np.ones(len(x))})
            name=CLEAN[source];before=base[name].to_numpy().copy()
            if self.variant=='V1':
                base[name]=np.where(valid,m,par['main_median'])
            elif self.variant in ['V2','V3','V4','V5']:
                lo,hi,step={'structure_age':(0,200,10),'storey_count':(1,7,1),'footprint_area':(50,1450,50),'f_165':(4,64,4)}[source]
                estimated=np.clip(np.round(latent/step)*step,lo,hi)
                # V2-V4 blend valid main measurements mildly; V5 never does.
                clean=np.where(valid,m if self.variant=='V5' else .9*m+.1*np.clip(latent,lo,hi),estimated)
                if source=='footprint_area':
                    ten=(m>=1500)&(np.remainder(m,500)==0)&(abs(m/10-latent)<abs(m-latent))
                    clean=np.where(ten,m/10,clean)
                if source=='f_165':clean=np.where((m<0)&(m>=-64),abs(m),clean)
                base[name]=clean
            repairs.append(dict(source=source,variant=self.variant,changed=int(np.sum(~np.isclose(before,base[name],equal_nan=True))),
                                untrusted=int((~valid).sum()),examples=x.loc[~np.isclose(before,base[name],equal_nan=True),'id'].head(8).tolist()))
            if self.variant!='V0':
                proxy_name={'structure_age':'age_proxy_mean','storey_count':'storey_proxy_mean','footprint_area':'area_proxy_mean','f_165':'height_proxy_mean'}[source]
                base[proxy_name]=latent if self.variant!='V1' else par['main_median']
        if self.variant!='V0':
            base['age_blended']=base.clean_age if self.variant in ['V1','V5'] else .8*base.clean_age+.2*base.age_proxy_mean.clip(0,120)
            base['storey_blended']=base.clean_storey if self.variant in ['V1','V5'] else .8*base.clean_storey+.2*base.storey_proxy_mean.clip(1,7)
            base['height_blended']=base.clean_height
            base['log_clean_age']=np.log1p(base.clean_age.clip(lower=0));base['log_clean_area']=np.log1p(base.clean_area)
        self.last_repairs=repairs
        return base,pd.DataFrame(uncert,index=x.index)
