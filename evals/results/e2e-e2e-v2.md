# End-to-end report: e2e-v2

## KPIs

| KPI | Value |
| --- | --- |
| Tickets | 200 |
| Intent accuracy | 0.9950 |
| KB hit rate@4 | 0.9100 |
| Grounded | 0.9400 |
| Escalation rate | 0.1300 |
| Accuracy on non-escalated tickets | 0.9943 |
| Wrong classifications escalated (recall on errors) | 0.0000 |
| LLM cost of the original uncached run (USD) | $0.0393 |

Note: this report was re-scored from cached replies after the citation-splitter fix, so the re-score itself cost $0.

## Escalation reasons

| Reason | Tickets | Rate |
| --- | --- | --- |
| not_grounded | 12 | 0.0600 |
| human_required_intent | 10 | 0.0500 |
| angry_or_offensive_language | 3 | 0.0150 |
| llm_knn_disagreement | 1 | 0.0050 |
