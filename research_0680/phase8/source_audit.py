"""Executable audit of immutable supplied files; never execute their training helpers."""
import ast,hashlib,importlib.util,json,tempfile,subprocess,sys,shutil,os
from pathlib import Path
import numpy as np
import pandas as pd
from research_0680.src.common import read_data,dump,ROOT

HERE=Path(__file__).resolve().parent

def provided():
    spec=importlib.util.spec_from_file_location('provided_phase8',HERE/'evidence/optimize_pipeline_4.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    out=Path(os.environ.get('RESEARCH0680_PHASE8_AUDIT_RESULTS',str(HERE/'audit_results')))
    if out.exists() and any(out.iterdir()):raise SystemExit('Preserve existing audit results')
    out.mkdir(parents=True,exist_ok=True);m=provided();x,y,test,_=read_data();base=m.clean_and_engineer_base_features(x,test)
    tree=ast.parse((HERE/'evidence/optimize_pipeline_4.py').read_text());entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=sorted(set(ast.unparse(n.func) for n in ast.walk(entry) if isinstance(n,ast.Call)))
    dummy_te=pd.DataFrame({f'{prefix}_bal_p{k}':np.full(len(base),1/3) for prefix in ['te_f90','te_f109','te_f141_m10'] for k in [1,2,3]})
    try:m.compute_clean_preserving_blup_and_sandwich_features(base,dummy_te.to_numpy(),dummy_te.columns.tolist())
    except KeyError as e:key_error=str(e)
    else:raise AssertionError('Expected supplied schema mismatch')
    # Execute only the TE transformer with source fit/apply indices, no classifiers.
    tr=np.arange(12000);va=np.arange(12000,15000);onehot=np.eye(3)[y]
    te=m.fit_apply_te_and_res(base,onehot,y+1,np.zeros((len(base),1)),tr,va,include_seismic_res=False)
    matches=[c for c in te if c.endswith('_p3') and any(v in c for v in ['f_141','f_109','f_90'])]
    compatible=base.iloc[va].copy();compatible['clean_storeys']=compatible.clean_storey
    broken=m.compute_clean_preserving_blup_and_sandwich_features(compatible,te.to_numpy(),te.columns.tolist())
    from .fixes import SandwichFeatures
    # Here only a demonstration of the correction responding to actual TE. The
    # paired experiment separately fits normalization strictly on training rows.
    fixed=SandwichFeatures().fit(compatible,te).transform(compatible,te)
    rng=np.random.default_rng(13);a=rng.dirichlet(np.ones(3),size=30);t=rng.dirichlet(np.ones(3),size=12);b=t[:,::-1]
    m.enforce_exact_cardinality=lambda p,counts:p.argmax(1)+1  # isolate mixing, not quota behavior
    o0,p0,_=m.run_v11_dual_plan_inference(a,t,b,a,t,b,w_planB=0)
    o1,p1,_=m.run_v11_dual_plan_inference(a,t,b,a,t,b,w_planB=1)
    # Quota is explicitly stubbed in this mixing-only demonstration.
    dump(out/'source_findings.json',dict(main_call_names=calls,base_shape=list(base.shape),schema_error=key_error,te_feature_count=len(te.columns),original_geographic_matches=matches,original_typeA_nonzero=int(np.count_nonzero(broken.iloc[:,0])),original_typeB_nonzero=int(np.count_nonzero(broken.iloc[:,1])),fixed_typeA_nonzero=int(np.count_nonzero(fixed[:,0])),fixed_typeB_nonzero=int(np.count_nonzero(fixed[:,1])),planB_oof_max_change=float(abs(o0-o1).max()),planB_test_max_change=float(abs(p0-p1).max())))

if __name__=='__main__':main()
