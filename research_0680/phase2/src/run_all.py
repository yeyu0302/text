"""Phase 2 reproducible entry point; environment overrides preserve shipped results."""
from .audit_claims import run as audit
from .run import screening, run_final
from .finalize import reconstruction, finalize
from .report import run as report

if __name__=='__main__':
    audit();screening();run_final();reconstruction();finalize();report()
