import numpy as np
import pandas as pd
from research_0680.src.common import read_data

def validate(sub,sample):
    assert list(sub)==['id','label'] and sub.id.equals(sample.id)
    assert len(sub)==5100 and sub.id.is_unique and sub.label.isin([1,2,3]).all() and not sub.isna().any().any()

def test_schema():
    _,_,_,s=read_data();a=pd.DataFrame(dict(id=s.id,label=np.ones(len(s),int)));validate(a,s)
