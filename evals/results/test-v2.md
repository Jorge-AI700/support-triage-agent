# Classification report: test-v2

Models: deepseek/deepseek-v4.1-flash-1017

## Overall

| Metric | Value |
| --- | --- |
| Tickets | 5419 |
| Intent accuracy | 0.9963 |
| Category accuracy | 0.9994 |
| Intent macro-F1 | 0.9963 |
| Category macro-F1 | 0.9995 |

## Per-intent metrics

| Intent | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| cancel_order | 1.0000 | 1.0000 | 1.0000 | 208 |
| change_order | 1.0000 | 1.0000 | 1.0000 | 214 |
| change_shipping_address | 1.0000 | 0.9949 | 0.9974 | 195 |
| check_cancellation_fee | 1.0000 | 1.0000 | 1.0000 | 190 |
| check_invoice | 1.0000 | 0.9900 | 0.9950 | 200 |
| check_payment_methods | 1.0000 | 1.0000 | 1.0000 | 200 |
| check_refund_policy | 1.0000 | 1.0000 | 1.0000 | 199 |
| complaint | 1.0000 | 1.0000 | 1.0000 | 200 |
| contact_customer_service | 1.0000 | 1.0000 | 1.0000 | 200 |
| contact_human_agent | 0.9950 | 1.0000 | 0.9975 | 200 |
| create_account | 0.9948 | 0.9796 | 0.9871 | 196 |
| delete_account | 1.0000 | 1.0000 | 1.0000 | 197 |
| delivery_options | 1.0000 | 1.0000 | 1.0000 | 196 |
| delivery_period | 1.0000 | 1.0000 | 1.0000 | 200 |
| edit_account | 1.0000 | 1.0000 | 1.0000 | 200 |
| get_invoice | 0.9903 | 1.0000 | 0.9951 | 205 |
| get_refund | 0.9595 | 0.9953 | 0.9771 | 214 |
| newsletter_subscription | 1.0000 | 1.0000 | 1.0000 | 200 |
| payment_issue | 1.0000 | 1.0000 | 1.0000 | 200 |
| place_order | 1.0000 | 0.9950 | 0.9975 | 200 |
| recover_password | 1.0000 | 0.9950 | 0.9975 | 199 |
| registration_problems | 0.9851 | 0.9950 | 0.9900 | 200 |
| review | 1.0000 | 1.0000 | 1.0000 | 199 |
| set_up_shipping_address | 0.9950 | 1.0000 | 0.9975 | 199 |
| switch_account | 0.9952 | 1.0000 | 0.9976 | 207 |
| track_order | 0.9952 | 1.0000 | 0.9976 | 209 |
| track_refund | 0.9946 | 0.9531 | 0.9734 | 192 |

## Top 10 confusions

| True intent | Predicted intent | Count |
| --- | --- | --- |
| track_refund | get_refund | 9 |
| create_account | registration_problems | 3 |
| change_shipping_address | set_up_shipping_address | 1 |
| check_invoice | contact_human_agent | 1 |
| check_invoice | get_invoice | 1 |
| create_account | switch_account | 1 |
| get_refund | track_refund | 1 |
| place_order | get_invoice | 1 |
| recover_password | track_order | 1 |
| registration_problems | create_account | 1 |

## Accuracy by confidence bucket

| Confidence bucket | Count | Accuracy | Mean confidence |
| --- | --- | --- | --- |
| 0.0-0.2 | 7 | 1.0000 | 0.0000 |
| 0.2-0.4 | 0 | — | — |
| 0.4-0.6 | 9 | 0.4444 | 0.5889 |
| 0.6-0.8 | 14 | 1.0000 | 0.7657 |
| 0.8-1.0 | 5389 | 0.9972 | 0.9744 |
