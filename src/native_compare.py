from src.train import run

if __name__=='__main__':
    native=run('C03_native_cat','full','cat',native=True,overrides={'thread_count':1,'max_ctr_complexity':2,'iterations':500,'learning_rate':.05})
    # Only expand the costly categorical branch if it beats the observed
    # one-hot CatBoost by a material amount. Same budget in this paired test.
    import json
    from src.train import ROOT
    ordinary=json.loads((ROOT/'results/predictions/C01_full_cat/summary.json').read_text())
    if native['balanced_accuracy'] > ordinary['balanced_accuracy']+.001:
        run('C04_native_no_te','proxy','cat',native=True,overrides={'thread_count':1,'max_ctr_complexity':2,'iterations':500,'learning_rate':.05})
    run('C05_core_xgb','core','xgb',overrides={'n_jobs':1})
    run('C06_core_hist','core','hist')
    run('C07_core_spline','core','spline')
