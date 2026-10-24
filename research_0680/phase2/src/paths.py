import os
from pathlib import Path
from research_0680.src.common import PROJECT,RESULTS
PHASE=PROJECT/'phase2'
OUT=Path(os.environ.get('RESEARCH0680_PHASE2_RESULTS',str(PHASE/'results'))).resolve()
REPORTS=Path(os.environ.get('RESEARCH0680_PHASE2_REPORTS',str(PHASE/'reports'))).resolve()
REFERENCE_RESULTS=RESULTS
