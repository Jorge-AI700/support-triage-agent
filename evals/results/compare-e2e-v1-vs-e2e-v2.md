# E2E comparison: e2e-v1 vs e2e-v2

| KPI | Before (e2e-v1) | After (e2e-v2) | Delta |
| --- | --- | --- | --- |
| Tickets | 200 | 200 | +0 |
| Intent accuracy | 0.9000 | 0.9950 | +0.0950 |
| KB hit rate@4 | 0.9100 | 0.9100 | +0.0000 |
| Grounded | 0.8800 | 0.9400 | +0.0600 |
| Escalation rate | 0.1850 | 0.1300 | -0.0550 |
| Accuracy on non-escalated tickets | 0.9264 | 0.9943 | +0.0679 |
| Wrong classifications escalated (recall on errors) | 0.4000 | 0.0000 | -0.4000 |
| LLM cost for the run (USD) | 0.0337 | 0.0393 | +0.0056 |


Notes: costs are from the original uncached runs (the re-score after the citation-splitter fix reused cached replies at $0). Recall on errors for v2 is 0/1: only one ticket was misclassified, so that rate is not meaningful at this sample size.
