# Classification report: sample-v1

Models: deepseek/deepseek-v4.1-flash-1017

## Overall

| Metric | Value |
| --- | --- |
| Tickets | 300 |
| Intent accuracy | 0.9100 |
| Category accuracy | 0.9633 |
| Intent macro-F1 | 0.9118 |
| Category macro-F1 | 0.9605 |

## Per-intent metrics

| Intent | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| cancel_order | 0.9091 | 1.0000 | 0.9524 | 10 |
| change_order | 1.0000 | 1.0000 | 1.0000 | 10 |
| change_shipping_address | 0.9444 | 0.9444 | 0.9444 | 18 |
| check_cancellation_fee | 1.0000 | 0.8571 | 0.9231 | 14 |
| check_invoice | 1.0000 | 0.5455 | 0.7059 | 11 |
| check_payment_methods | 1.0000 | 1.0000 | 1.0000 | 8 |
| check_refund_policy | 0.8571 | 0.8571 | 0.8571 | 14 |
| complaint | 1.0000 | 1.0000 | 1.0000 | 7 |
| contact_customer_service | 0.7273 | 1.0000 | 0.8421 | 16 |
| contact_human_agent | 1.0000 | 0.8571 | 0.9231 | 14 |
| create_account | 1.0000 | 0.9333 | 0.9655 | 15 |
| delete_account | 1.0000 | 0.8750 | 0.9333 | 8 |
| delivery_options | 1.0000 | 0.6667 | 0.8000 | 9 |
| delivery_period | 1.0000 | 0.8750 | 0.9333 | 8 |
| edit_account | 0.6667 | 1.0000 | 0.8000 | 8 |
| get_invoice | 0.6875 | 1.0000 | 0.8148 | 11 |
| get_refund | 1.0000 | 1.0000 | 1.0000 | 9 |
| newsletter_subscription | 1.0000 | 1.0000 | 1.0000 | 11 |
| payment_issue | 1.0000 | 1.0000 | 1.0000 | 11 |
| place_order | 0.7857 | 0.9167 | 0.8462 | 12 |
| recover_password | 1.0000 | 1.0000 | 1.0000 | 8 |
| registration_problems | 0.9091 | 1.0000 | 0.9524 | 10 |
| review | 1.0000 | 0.8750 | 0.9333 | 16 |
| set_up_shipping_address | 0.8889 | 0.8889 | 0.8889 | 9 |
| switch_account | 1.0000 | 0.5833 | 0.7368 | 12 |
| track_order | 0.8750 | 1.0000 | 0.9333 | 7 |
| track_refund | 0.8750 | 1.0000 | 0.9333 | 14 |

## Top 10 confusions

| True intent | Predicted intent | Count |
| --- | --- | --- |
| check_invoice | get_invoice | 5 |
| switch_account | edit_account | 4 |
| delivery_options | place_order | 3 |
| check_cancellation_fee | check_refund_policy | 2 |
| check_refund_policy | track_refund | 2 |
| contact_human_agent | contact_customer_service | 2 |
| review | contact_customer_service | 2 |
| change_shipping_address | set_up_shipping_address | 1 |
| create_account | registration_problems | 1 |
| delete_account | cancel_order | 1 |

## Accuracy by confidence bucket

| Confidence bucket | Count | Accuracy | Mean confidence |
| --- | --- | --- | --- |
| 0.0-0.2 | 0 | — | — |
| 0.2-0.4 | 4 | 0.5000 | 0.4000 |
| 0.4-0.6 | 16 | 0.5625 | 0.5813 |
| 0.6-0.8 | 20 | 0.8000 | 0.7460 |
| 0.8-1.0 | 260 | 0.9462 | 0.9329 |
