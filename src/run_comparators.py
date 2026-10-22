from src.train import run

if __name__=='__main__':
    # One CPU thread while the four-thread ablation job runs on this 5-CPU host.
    run('C00_full_lgb','full','lgb',overrides={'n_jobs':1})
    run('C01_full_cat','full','cat',overrides={'thread_count':1})
    run('C02_full_logistic','full','logistic')
