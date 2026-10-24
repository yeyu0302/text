"""Portable second-round notebook and GitHub download bundle."""
import hashlib,json,zipfile
from research_0680.src.common import ROOT,PROJECT,dump
from .paths import PHASE

def notebook():
    nb=json.loads((PROJECT/'notebooks/Kaggle_0680_Colab_Reproducible.ipynb').read_text())
    def put(i,text):nb['cells'][i]['source']=text.splitlines(True)
    put(0,'''# 五項困難：第二輪可重現實驗
本輪已測得的條件村落模型未通過兩種子門檻；不保證Kaggle0.680。此Notebook會重新訓練，並將結果寫到新目錄，保留既有成果。

請從GitHub分支 `research-0680-20261024` 的 Code → Download ZIP 下載完整儲存庫，再於下面上傳。私人儲存庫先在瀏覽器登入下載，不需要Token。如果使用 `Kaggle_Phase2_complete.zip` 交付包，請同時上傳原始 `train.csv`、`test.csv`、`sample_submission.csv`。
''')
    env=''.join(nb['cells'][2]['source']).replace("/'colab_runs'/","/'phase2_colab_runs'/")
    env=env.replace("os.environ['MPLCONFIGDIR']", "os.environ['RESEARCH0680_PHASE2_RESULTS']=str(run_dir/'phase2_results')\nos.environ['RESEARCH0680_PHASE2_REPORTS']=str(run_dir/'phase2_reports')\nos.environ['MPLCONFIGDIR']")
    put(2,env)
    put(3,'''## 完整重跑
執行說法核對、九項開發消融、兩種子外層與內層選模、量測重建及提交門檻。新輸出目錄沒有參考OOF時會重新計算固定參考。CPU可能需數分鐘至數十分鐘。
所有最終λ與混合權重僅由外層訓練資料內層選擇；開發集最高值不是Kaggle分數。使用者0.679原程式目前不可原樣重現，本參考OOF不得冒稱其OOF。
''')
    put(4,"subprocess.check_call([sys.executable,'-B','-m','research_0680.phase2.src.run_all'],cwd=repo,env=os.environ.copy())\n")
    put(5,"print((run_dir/'phase2_results/nested_comparison.csv').read_text())\nprint((run_dir/'phase2_results/submission_gate.json').read_text())\nprint((run_dir/'phase2_reports/final_report.md').read_text())\n")
    put(6,'''## 下載
只有兩種子都通過ΔBA>.001且paired95%CI下界>0，才建立新候選。既有結果未通過，所以沒有新提交。資料、程式、CSV、指標與紀錄會包含在下列重跑ZIP。
''')
    put(7,''.join(nb['cells'][7]['source']).replace('Kaggle_0680_reproduction_results.zip','Kaggle_Phase2_reproduction_results.zip'))
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<colab-cell>','exec')
    folder=PHASE/'notebooks';folder.mkdir(exist_ok=True)
    (folder/'Kaggle_Phase2_Colab.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))

def bundle():
    folder=PHASE/'downloads';folder.mkdir(exist_ok=True)
    target=folder/'Kaggle_Phase2_complete.zip'
    assert not target.exists(),'Preserve previous delivery; create a separately named release'
    paths=[p for p in PHASE.rglob('*') if p.is_file() and not p.is_symlink() and 'downloads' not in p.relative_to(PHASE).parts and '__pycache__' not in p.parts]
    for sub in ['src','configs','tests','docs']:
        paths.extend(p for p in (PROJECT/sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend([PROJECT/'__init__.py',PROJECT/'README.md',PROJECT/'notebooks/Kaggle_0680_Colab_Reproducible.ipynb'])
    paths.extend([ROOT/'src/__init__.py',ROOT/'src/features.py',ROOT/'src/evaluation.py',ROOT/'optimize_pipeline.py',ROOT/'results/reproducibility.json'])
    for seed in [42,2026]:
        paths.extend(p for p in (PROJECT/'results/00_baseline'/f'fixed_baseline_seed{seed}').glob('*') if p.is_file() and p.suffix!='.npz')
    paths.extend([PROJECT/'results/00_baseline/provided'/f for f in ['submission_user_0679.csv','optimize_pipeline_user.py','best_params_user.json']])
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(paths)):z.write(p,'text/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in sorted(set(paths)):
            assert z.read('text/'+str(p.relative_to(ROOT)))==p.read_bytes()
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    (folder/(target.name+'.sha256')).write_text(sha+'  '+target.name+'\n')
    dump(folder/'bundle_inventory.json',dict(bytes=target.stat().st_size,sha256=sha,
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths))],
        data_note='Original train/test/sample CSV excluded; GitHub repository ZIP contains them, or upload them alongside this bundle.'))
    print(target,target.stat().st_size,sha)

if __name__=='__main__':notebook();bundle()
