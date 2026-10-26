# Resultados: BANKING77 con Strands Decider

Modelo servido: strands-decider-2B-hobson-v21. Ejemplos de prueba: 770.

La latencia es la que reporta el servidor del decider, sin red ni LiteLLM.

## Resumen

| condicion                  |   accuracy |   macro_f1 |    top3 |    top5 |     ECE |   latencia_p50_ms |   latencia_p95_ms |
|:---------------------------|-----------:|-----------:|--------:|--------:|--------:|------------------:|------------------:|
| TF-IDF + SVM (supervisado) |      0.916 |      0.915 | nan     | nan     | nan     |           nan     |           nan     |
| Decider: descriptions      |      0.801 |      0.798 |   0.953 |   0.974 |   0.205 |           211.925 |           214.856 |
| Decider: names             |      0.843 |      0.839 |   0.962 |   0.984 |   0.034 |            83.140 |            85.969 |
| Decider: names_shuffled    |      0.842 |      0.840 |   0.964 |   0.987 |   0.041 |            84.020 |            86.877 |

## Condición: descriptions

Tokens de entrada: mediana 1447, máximo 1493.

### Exactitud según umbral de probabilidad

|   umbral |   cobertura |   accuracy_aceptadas |
|---------:|------------:|---------------------:|
|    0.000 |       1.000 |                0.801 |
|    0.300 |       0.904 |                0.838 |
|    0.500 |       0.653 |                0.918 |
|    0.700 |       0.358 |                0.964 |
|    0.800 |       0.199 |                0.987 |
|    0.900 |       0.078 |                0.983 |
|    0.950 |       0.005 |                0.750 |

### Diez intenciones con peor F1

| intencion                          |   precision |   recall |   f1-score |
|:-----------------------------------|------------:|---------:|-----------:|
| card_delivery_estimate             |       1.000 |    0.100 |      0.182 |
| get_physical_card                  |       0.429 |    0.300 |      0.353 |
| topping_up_by_card                 |       0.333 |    0.400 |      0.364 |
| top_up_by_card_charge              |       1.000 |    0.300 |      0.462 |
| supported_cards_and_currencies     |       0.438 |    0.700 |      0.538 |
| beneficiary_not_allowed            |       1.000 |    0.400 |      0.571 |
| transfer_not_received_by_recipient |       1.000 |    0.400 |      0.571 |
| transfer_timing                    |       0.545 |    0.600 |      0.571 |
| order_physical_card                |       0.714 |    0.500 |      0.588 |
| failed_transfer                    |       0.471 |    0.800 |      0.593 |

### Confusiones más frecuentes

| real                                    | predicha                       |   veces |
|:----------------------------------------|:-------------------------------|--------:|
| card_delivery_estimate                  | card_arrival                   |       7 |
| get_physical_card                       | change_pin                     |       7 |
| top_up_by_card_charge                   | supported_cards_and_currencies |       5 |
| get_disposable_virtual_card             | disposable_card_limits         |       4 |
| top_up_by_bank_transfer_charge          | transfer_fee_charged           |       4 |
| beneficiary_not_allowed                 | failed_transfer                |       4 |
| balance_not_updated_after_bank_transfer | pending_transfer               |       4 |
| virtual_card_not_working                | disposable_card_limits         |       4 |
| fiat_currency_support                   | exchange_via_app               |       3 |
| order_physical_card                     | get_physical_card              |       3 |
| direct_debit_payment_not_recognised     | card_payment_not_recognised    |       3 |
| why_verify_identity                     | unable_to_verify_identity      |       3 |

## Condición: names

Tokens de entrada: mediana 625, máximo 671.

### Exactitud según umbral de probabilidad

|   umbral |   cobertura |   accuracy_aceptadas |
|---------:|------------:|---------------------:|
|    0.000 |       1.000 |                0.843 |
|    0.300 |       0.991 |                0.849 |
|    0.500 |       0.923 |                0.878 |
|    0.700 |       0.748 |                0.929 |
|    0.800 |       0.629 |                0.967 |
|    0.900 |       0.504 |                0.985 |
|    0.950 |       0.395 |                0.987 |

### Diez intenciones con peor F1

| intencion                               |   precision |   recall |   f1-score |
|:----------------------------------------|------------:|---------:|-----------:|
| topping_up_by_card                      |       0.500 |    0.300 |      0.375 |
| order_physical_card                     |       0.600 |    0.300 |      0.400 |
| extra_charge_on_statement               |       0.625 |    0.500 |      0.556 |
| card_delivery_estimate                  |       1.000 |    0.400 |      0.571 |
| balance_not_updated_after_bank_transfer |       0.714 |    0.500 |      0.588 |
| supported_cards_and_currencies          |       0.667 |    0.600 |      0.632 |
| pending_transfer                        |       0.533 |    0.800 |      0.640 |
| get_physical_card                       |       0.533 |    0.800 |      0.640 |
| card_payment_not_recognised             |       0.571 |    0.800 |      0.667 |
| fiat_currency_support                   |       0.636 |    0.700 |      0.667 |

### Confusiones más frecuentes

| real                                    | predicha                       |   veces |
|:----------------------------------------|:-------------------------------|--------:|
| card_delivery_estimate                  | card_arrival                   |       6 |
| balance_not_updated_after_bank_transfer | pending_transfer               |       4 |
| order_physical_card                     | get_physical_card              |       4 |
| why_verify_identity                     | unable_to_verify_identity      |       3 |
| pending_top_up                          | top_up_failed                  |       3 |
| direct_debit_payment_not_recognised     | card_payment_not_recognised    |       3 |
| transfer_fee_charged                    | top_up_by_bank_transfer_charge |       3 |
| request_refund                          | cancel_transfer                |       2 |
| pending_transfer                        | transfer_timing                |       2 |
| reverted_card_payment?                  | declined_card_payment          |       2 |
| supported_cards_and_currencies          | fiat_currency_support          |       2 |
| topping_up_by_card                      | pending_top_up                 |       2 |

## Condición: names_shuffled

Tokens de entrada: mediana 625, máximo 671.

### Exactitud según umbral de probabilidad

|   umbral |   cobertura |   accuracy_aceptadas |
|---------:|------------:|---------------------:|
|    0.000 |       1.000 |                0.842 |
|    0.300 |       0.996 |                0.844 |
|    0.500 |       0.922 |                0.882 |
|    0.700 |       0.764 |                0.922 |
|    0.800 |       0.660 |                0.959 |
|    0.900 |       0.509 |                0.985 |
|    0.950 |       0.404 |                1.000 |

### Diez intenciones con peor F1

| intencion                               |   precision |   recall |   f1-score |
|:----------------------------------------|------------:|---------:|-----------:|
| topping_up_by_card                      |       0.500 |    0.300 |      0.375 |
| order_physical_card                     |       0.667 |    0.400 |      0.500 |
| transfer_not_received_by_recipient      |       0.600 |    0.600 |      0.600 |
| get_physical_card                       |       0.538 |    0.700 |      0.609 |
| supported_cards_and_currencies          |       0.538 |    0.700 |      0.609 |
| extra_charge_on_statement               |       0.667 |    0.600 |      0.632 |
| fiat_currency_support                   |       0.636 |    0.700 |      0.667 |
| balance_not_updated_after_bank_transfer |       0.750 |    0.600 |      0.667 |
| card_payment_not_recognised             |       0.700 |    0.700 |      0.700 |
| card_swallowed                          |       0.857 |    0.600 |      0.706 |

### Confusiones más frecuentes

| real                                    | predicha                           |   veces |
|:----------------------------------------|:-----------------------------------|--------:|
| balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient |       4 |
| card_acceptance                         | supported_cards_and_currencies     |       4 |
| card_delivery_estimate                  | card_arrival                       |       3 |
| pending_top_up                          | top_up_failed                      |       3 |
| pending_transfer                        | transfer_timing                    |       3 |
| topping_up_by_card                      | top_up_reverted                    |       3 |
| order_physical_card                     | get_physical_card                  |       3 |
| fiat_currency_support                   | exchange_via_app                   |       2 |
| order_physical_card                     | card_arrival                       |       2 |
| card_swallowed                          | get_physical_card                  |       2 |
| get_physical_card                       | change_pin                         |       2 |
| Refund_not_showing_up                   | request_refund                     |       2 |
