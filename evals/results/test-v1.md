# Classification report: test-v1

Models: deepseek/deepseek-v4.1-flash-1017

## Overall

| Metric | Value |
| --- | --- |
| Tickets | 5419 |
| Intent accuracy | 0.9105 |
| Category accuracy | 0.9611 |
| Intent macro-F1 | 0.9091 |
| Category macro-F1 | 0.8808 |

## Per-intent metrics

| Intent | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| cancel_order | 0.9535 | 0.9856 | 0.9693 | 208 |
| change_order | 1.0000 | 0.9860 | 0.9929 | 214 |
| change_shipping_address | 0.9462 | 0.9026 | 0.9239 | 195 |
| check_cancellation_fee | 1.0000 | 0.9000 | 0.9474 | 190 |
| check_invoice | 0.9554 | 0.5350 | 0.6859 | 200 |
| check_payment_methods | 0.9804 | 1.0000 | 0.9901 | 200 |
| check_refund_policy | 0.8863 | 0.9397 | 0.9122 | 199 |
| complaint | 0.9272 | 0.9550 | 0.9409 | 200 |
| contact_customer_service | 0.7729 | 0.9700 | 0.8603 | 200 |
| contact_human_agent | 0.9692 | 0.9450 | 0.9570 | 200 |
| create_account | 0.9630 | 0.9286 | 0.9455 | 196 |
| delete_account | 1.0000 | 0.9442 | 0.9713 | 197 |
| delivery_options | 1.0000 | 0.8622 | 0.9260 | 196 |
| delivery_period | 0.8168 | 0.8250 | 0.8209 | 200 |
| edit_account | 0.7692 | 1.0000 | 0.8696 | 200 |
| get_invoice | 0.6894 | 0.9854 | 0.8112 | 205 |
| get_refund | 0.8894 | 0.8645 | 0.8768 | 214 |
| newsletter_subscription | 0.9852 | 1.0000 | 0.9926 | 200 |
| payment_issue | 0.9801 | 0.9850 | 0.9825 | 200 |
| place_order | 0.8598 | 0.9200 | 0.8889 | 200 |
| recover_password | 1.0000 | 1.0000 | 1.0000 | 199 |
| registration_problems | 0.9209 | 0.9900 | 0.9542 | 200 |
| review | 0.9792 | 0.9447 | 0.9616 | 199 |
| set_up_shipping_address | 0.9265 | 0.9497 | 0.9380 | 199 |
| switch_account | 0.9752 | 0.5700 | 0.7195 | 207 |
| track_order | 0.8230 | 0.8230 | 0.8230 | 209 |
| track_refund | 0.8895 | 0.8802 | 0.8848 | 192 |

## Top 10 confusions

| True intent | Predicted intent | Count |
| --- | --- | --- |
| check_invoice | get_invoice | 91 |
| switch_account | edit_account | 56 |
| track_order | delivery_period | 37 |
| delivery_period | track_order | 35 |
| delivery_options | place_order | 27 |
| track_refund | get_refund | 23 |
| switch_account | contact_customer_service | 18 |
| check_cancellation_fee | check_refund_policy | 17 |
| change_shipping_address | set_up_shipping_address | 15 |
| get_refund | complaint | 13 |

## Accuracy by confidence bucket

| Confidence bucket | Count | Accuracy | Mean confidence |
| --- | --- | --- | --- |
| 0.0-0.2 | 5 | 0.4000 | 0.1200 |
| 0.2-0.4 | 80 | 0.3875 | 0.3900 |
| 0.4-0.6 | 469 | 0.7377 | 0.5865 |
| 0.6-0.8 | 262 | 0.8282 | 0.7471 |
| 0.8-1.0 | 4603 | 0.9424 | 0.9337 |
