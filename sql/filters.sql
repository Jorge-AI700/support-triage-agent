-- TODO: 1. Test-set tickets by intent
-- Filter the tickets table for test-set records (split = 'test') grouped or filtered by a specific intent.

-- TODO: 2. Misclassified tickets for a run
-- Join predictions with tickets for a given run_id where predictions.pred_intent != tickets.intent
-- or predictions.pred_category != tickets.category to inspect error cases.

-- TODO: 3. Escalation rate by category
-- Calculate the proportion and count of escalated tickets (escalated = true)
-- grouped by ticket category for a specific evaluation run.

-- TODO: 4. Accuracy per intent
-- Compute classification accuracy (count of matching predicted intents / total tickets)
-- grouped by ground-truth intent for a specific evaluation run.
