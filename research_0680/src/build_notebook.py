import json
from .common import PROJECT

def run():
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
    md('''# Kaggle 0.680 研究流程：可重現 Colab\n本 Notebook 執行 S0–S6、內層選模與外層評估。**不保證 Kaggle 0.680，也不會輸出未通過門檻的候選提交。**\n\n請先從 GitHub 分支 `research-0680-20261024` 的 Code → Download ZIP 下載完整儲存庫。若是私人儲存庫，先在瀏覽器登入 GitHub 下載，再於下面上傳；不需要在 Notebook 或聊天貼入 Token。完整儲存庫 ZIP 包含原始 train/test/sample 資料。若只下載研究交付包，請另行提供這三份 CSV。\n\n既有結果保持不變，每次執行使用獨立輸出目錄。''')
    code('''from pathlib import Path
import os,sys,subprocess,json,zipfile,time,shutil
base=Path.cwd()
matches=[p.parent for p in base.rglob('train.csv') if (p.parent/'research_0680'/'src').is_dir() and '.venvs' not in p.parts]
if not matches:
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError('請在既有儲存庫目錄執行，或在 Colab 上傳完整儲存庫 ZIP。')
    uploaded=files.upload()
    target=base/'earthquake_repository';target.mkdir(exist_ok=True)
    for name in uploaded:
        if name.endswith('.zip'):
            with zipfile.ZipFile(name) as archive:
                for item in archive.infolist():
                    destination=(target/item.filename).resolve()
                    if not destination.is_relative_to(target.resolve()):
                        raise ValueError('拒絕 ZIP 路徑越界')
                archive.extractall(target)
    for project in target.rglob('research_0680'):
        if not (project/'src').is_dir():continue
        for name in ['train.csv','test.csv','sample_submission.csv']:
            if name in uploaded and not (project.parent/name).exists():
                shutil.copyfile(base/name,project.parent/name)
    matches=[p.parent for p in target.rglob('train.csv') if (p.parent/'research_0680'/'src').is_dir()]
if not matches:
    raise RuntimeError('找不到同時含 train.csv 與 research_0680/src 的完整儲存庫。請使用 GitHub Code → Download ZIP。')
repo=matches[0].resolve()
assert all((repo/f).is_file() for f in ['train.csv','test.csv','sample_submission.csv'])
print('Repository:',repo)
''')
    code('''subprocess.check_call([sys.executable,'-m','pip','install','-r',str(repo/'research_0680/configs/requirements.txt')])
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
run_dir=repo/'research_0680'/'colab_runs'/time.strftime('%Y%m%d_%H%M%S')
run_dir.mkdir(parents=True,exist_ok=False)
os.environ['RESEARCH0680_RESULTS']=str(run_dir/'results')
os.environ['RESEARCH0680_REPORTS']=str(run_dir/'reports')
os.environ['MPLCONFIGDIR']=str(run_dir/'matplotlib_cache')
print('Fresh output directory:',run_dir)
''')
    md('''## 執行全部流程\nCPU 訓練可能需要數十分鐘。每折結果會保存；不要以開發切分最高值當作最終分數。上傳的 V6 原程式與 V5 JSON 存在版本／欄位不一致，這個缺口會保留在報告，不會冒充 0.679 的可重現 OOF。''')
    code('''subprocess.check_call([sys.executable,'-B','-m','research_0680.src.run_all'],cwd=repo,env=os.environ.copy())
''')
    code('''print((run_dir/'results/nested_rule_comparison.csv').read_text())
print((run_dir/'results/05_final_candidates/candidate_gate_decisions.csv').read_text())
print((run_dir/'reports/final_recommendation.md').read_text())
''')
    md('''## 下載成果\n若沒有方法通過預設的 ΔBA／信賴區間／第二種子門檻，就不會產生 `submission_candidate_*.csv`。這是驗證結果，不是缺漏。保留目前使用者回報 0.679 的原提交，新的 Kaggle 成績只能在競賽實際評分後確認。''')
    code('''bundle=base/'Kaggle_0680_reproduction_results.zip'
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in run_dir.rglob('*'):
        if p.is_file() and 'matplotlib_cache' not in p.parts:
            z.write(p,p.relative_to(run_dir))
print(bundle,bundle.stat().st_size)
try:
    from google.colab import files
    files.download(str(bundle))
except ImportError:
    print('本機結果位於上面列出的 ZIP 路徑。')
''')
    for i,cell in enumerate(cells):cell['id']=f'cell-{i:03}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)
    path=PROJECT/'notebooks/Kaggle_0680_Colab_Reproducible.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2))
    for cell in cells:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook-cell>','exec')
    print('Notebook syntax checked; Colab platform execution not claimed',flush=True)

if __name__=='__main__':run()
