import json
import numpy as np
import pytest
from research_0680.src.common import read_data
from research_0680.phase3 import run,balanced,interactions,repair

@pytest.mark.parametrize('backend,protocol',[(run,'protocol.json'),(balanced,'balanced_protocol.json'),(interactions,'interaction_protocol.json'),(repair,'repair_protocol.json')])
def test_query_batch_cannot_change_fitted_model(backend,protocol,monkeypatch):
    """Unrelated query rows must not affect fitted scalers, priors or repair rules."""
    config=json.loads((run.HERE/protocol).read_text())
    monkeypatch.setattr(run,'PROTOCOL',config);monkeypatch.setattr(run,'CHOICES',config['choices'])
    x,y,_,_=read_data();idx=np.random.default_rng(91).permutation(len(x))
    fit=x.iloc[idx[:360]].copy();target=y[idx[:360]];query=x.iloc[idx[360:380]].copy()
    first=backend.predictions(fit,target,query.iloc[:8])
    second=backend.predictions(fit,target,query)
    for name in config['choices']:
        assert np.isfinite(first[name]).all()
        assert np.allclose(first[name].sum(1),1)
        assert np.max(abs(first[name]-second[name][:8]))<1e-10
