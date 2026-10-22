# Reproduce the audit and experiments

The original three project files and data remain unchanged. Existing submissions
are preserved. New outputs are under `results/`; no Kaggle upload is automated.
The provided public score 0.676 is the external benchmark, not a newly measured
result. See `results/final_report.md` for measured findings and scope.

Python 3.12, Linux CPU. Use the existing isolated checkout; no worktree is needed.
The setup never assumes `/content/` exists. From the repository root:

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
uv venv --allow-existing --python python3.12 /workspace/.venvs/text-research
uv pip install --python /workspace/.venvs/text-research/bin/python -r configs/requirements-research.lock
source /workspace/.venvs/text-research/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/workspace/.cache/matplotlib
python -B -m pytest -q
python -B -m src.audit
python -B -m src.screen
python -B -m src.leakage_probe
python -B -u -m src.run_ablation
python -B -u -m src.run_comparators
python -B -u -m src.corrected_controls
python -B -u -m src.refine
python -B -u -m src.native_compare
python -B -u -m src.finish_research
```

These are the sequential commands for a fresh experiment output set. Existing
completed experiment IDs are cached; final delivery refuses to overwrite an
existing primary submission. Keep the supplied frozen selection configs when
reproducing this report. The independent standard-optimizer audit controls are:

```bash
python -B -u -m src.train --name C08_core_standard_spline --family core --model spline_standard
python -B -u -m src.train --name R2026_C08_core_standard_spline --family core --model spline_standard --seed 2026
```

Reproduce the original two-model pipeline in the **minimal** onboarding venv,
which has only NumPy/pandas/SciPy and cannot activate optional models implicitly:

```bash
uv venv --allow-existing --python python3.12 /workspace/.venvs/text
uv pip install --python /workspace/.venvs/text/bin/python numpy==2.3.5 pandas==2.2.3 scipy==1.17.0
/workspace/.venvs/text/bin/python -B -u -m src.reproduce
```

Experiment names identify complete configurations. Completed runs are cached
and never silently overwritten; use a new name for a code or parameter change.
Each experiment saves configurations, code hash, OOF and test probabilities,
outer-fold metrics and runtime under `results/predictions/<experiment>/`.
Dataset/source hashes and package versions are in `results/reproducibility.json`.
`configs/experiments.json` defines model defaults. Exact commands for selected
follow-up experiments are recorded in the final report and selection config.

For an individual controlled run:

```bash
python -B -u -m src.train --name new_id --family full --model xgb --seed 42
```

No held-out labels are passed to preprocessing or estimator fitting. Fixed tree
budgets avoid using outer validation for early stopping. Calibration and fitted
ensemble rules must use `src.postprocess`, not a weight search on globally
generated OOF predictions. Review the leakage audit before extending the code.

The final decision solver optimizes a transportation LP and verifies exact class
counts. The competition PDF documents 1,700 observations per test class. This
assumption does not generalize to arbitrary replacement test datasets.
