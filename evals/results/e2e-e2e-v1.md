# End-to-end report: e2e-v1

## KPIs

| KPI | Value |
| --- | --- |
| Tickets | 200 |
| Intent accuracy | 0.9000 |
| KB hit rate@4 | 0.9100 |
| Grounded | 0.8800 |
| Escalation rate | 0.1850 |
| Accuracy on non-escalated tickets | 0.9264 |
| Wrong classifications escalated (recall on errors) | 0.4000 |
| LLM cost of the original uncached run (USD) | $0.0337 |

Note: this report was re-scored from cached replies after the citation-splitter fix, so the re-score itself cost $0.

## Escalation reasons

| Reason | Tickets | Rate |
| --- | --- | --- |
| not_grounded | 24 | 0.1200 |
| human_required_intent | 8 | 0.0400 |
| low_confidence | 4 | 0.0200 |
| angry_or_offensive_language | 3 | 0.0150 |
