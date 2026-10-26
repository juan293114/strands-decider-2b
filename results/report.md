# Resultados: BANKING77 con Strands Decider

Modelo servido: strands-decider-2B-hobson-v21. Ejemplos de prueba: 3080.

La latencia es la que reporta el servidor del decider, sin red ni LiteLLM.

## Resumen

| condicion                  |   accuracy |   macro_f1 |    top3 |    top5 |     ECE |   latencia_p50_ms |   latencia_p95_ms |
|:---------------------------|-----------:|-----------:|--------:|--------:|--------:|------------------:|------------------:|
| TF-IDF + SVM (supervisado) |      0.913 |      0.913 | nan     | nan     | nan     |           nan     |           nan     |
| Decider: descriptions      |      0.811 |      0.810 |   0.955 |   0.977 |   0.209 |           212.990 |           216.310 |
| Decider: names             |      0.844 |      0.842 |   0.969 |   0.985 |   0.022 |            83.400 |            86.092 |
| Decider: names_shuffled    |      0.844 |      0.844 |   0.969 |   0.985 |   0.019 |            84.035 |            86.871 |

## Condición: descriptions

Tokens de entrada: mediana 1447, máximo 1514.

### Exactitud según umbral de probabilidad

|   umbral |   cobertura |   accuracy_aceptadas |
|---------:|------------:|---------------------:|
|    0.000 |       1.000 |                0.811 |
|    0.300 |       0.900 |                0.851 |
|    0.500 |       0.674 |                0.921 |
|    0.700 |       0.373 |                0.969 |
|    0.800 |       0.206 |                0.981 |
|    0.900 |       0.069 |                0.995 |
|    0.950 |       0.006 |                0.941 |

### Diez intenciones con peor F1

| intencion                      |   precision |   recall |   f1-score |
|:-------------------------------|------------:|---------:|-----------:|
| topping_up_by_card             |       0.436 |    0.425 |      0.430 |
| get_physical_card              |       0.514 |    0.475 |      0.494 |
| top_up_by_card_charge          |       0.875 |    0.350 |      0.500 |
| supported_cards_and_currencies |       0.394 |    0.700 |      0.505 |
| beneficiary_not_allowed        |       0.933 |    0.350 |      0.509 |
| card_delivery_estimate         |       0.938 |    0.375 |      0.536 |
| top_up_by_bank_transfer_charge |       0.633 |    0.475 |      0.543 |
| order_physical_card            |       0.750 |    0.525 |      0.618 |
| declined_card_payment          |       0.686 |    0.600 |      0.640 |
| transfer_timing                |       0.628 |    0.675 |      0.651 |

### Confusiones más frecuentes

| real                                | predicha                       |   veces |
|:------------------------------------|:-------------------------------|--------:|
| top_up_by_card_charge               | supported_cards_and_currencies |      21 |
| card_delivery_estimate              | card_arrival                   |      19 |
| get_disposable_virtual_card         | disposable_card_limits         |      18 |
| get_physical_card                   | change_pin                     |      18 |
| beneficiary_not_allowed             | failed_transfer                |      16 |
| order_physical_card                 | get_physical_card              |      13 |
| why_verify_identity                 | unable_to_verify_identity      |      13 |
| virtual_card_not_working            | disposable_card_limits         |      13 |
| top_up_by_bank_transfer_charge      | transfer_fee_charged           |      13 |
| declined_card_payment               | card_not_working               |      12 |
| fiat_currency_support               | exchange_via_app               |      10 |
| direct_debit_payment_not_recognised | card_payment_not_recognised    |      10 |

## Condición: names

Tokens de entrada: mediana 625, máximo 692.

### Exactitud según umbral de probabilidad

|   umbral |   cobertura |   accuracy_aceptadas |
|---------:|------------:|---------------------:|
|    0.000 |       1.000 |                0.844 |
|    0.300 |       0.991 |                0.849 |
|    0.500 |       0.922 |                0.879 |
|    0.700 |       0.762 |                0.925 |
|    0.800 |       0.660 |                0.947 |
|    0.900 |       0.524 |                0.975 |
|    0.950 |       0.399 |                0.982 |

### Diez intenciones con peor F1

| intencion                               |   precision |   recall |   f1-score |
|:----------------------------------------|------------:|---------:|-----------:|
| order_physical_card                     |       0.682 |    0.375 |      0.484 |
| topping_up_by_card                      |       0.708 |    0.425 |      0.531 |
| balance_not_updated_after_bank_transfer |       0.692 |    0.450 |      0.545 |
| supported_cards_and_currencies          |       0.600 |    0.600 |      0.600 |
| why_verify_identity                     |       0.947 |    0.450 |      0.610 |
| transfer_not_received_by_recipient      |       0.578 |    0.650 |      0.612 |
| get_physical_card                       |       0.523 |    0.850 |      0.648 |
| fiat_currency_support                   |       0.617 |    0.725 |      0.667 |
| card_swallowed                          |       0.917 |    0.550 |      0.688 |
| card_arrival                            |       0.596 |    0.850 |      0.701 |

### Confusiones más frecuentes

| real                                    | predicha                           |   veces |
|:----------------------------------------|:-----------------------------------|--------:|
| order_physical_card                     | get_physical_card                  |      18 |
| card_delivery_estimate                  | card_arrival                       |      15 |
| why_verify_identity                     | unable_to_verify_identity          |      11 |
| why_verify_identity                     | verify_my_identity                 |      11 |
| balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient |      11 |
| direct_debit_payment_not_recognised     | card_payment_not_recognised        |       9 |
| supported_cards_and_currencies          | fiat_currency_support              |       9 |
| card_swallowed                          | get_physical_card                  |       8 |
| transfer_fee_charged                    | top_up_by_bank_transfer_charge     |       8 |
| fiat_currency_support                   | exchange_via_app                   |       8 |
| beneficiary_not_allowed                 | failed_transfer                    |       8 |
| declined_transfer                       | declined_card_payment              |       8 |

## Condición: names_shuffled

Tokens de entrada: mediana 625, máximo 692.

### Exactitud según umbral de probabilidad

|   umbral |   cobertura |   accuracy_aceptadas |
|---------:|------------:|---------------------:|
|    0.000 |       1.000 |                0.844 |
|    0.300 |       0.993 |                0.849 |
|    0.500 |       0.926 |                0.878 |
|    0.700 |       0.776 |                0.924 |
|    0.800 |       0.685 |                0.948 |
|    0.900 |       0.534 |                0.970 |
|    0.950 |       0.416 |                0.981 |

### Diez intenciones con peor F1

| intencion                               |   precision |   recall |   f1-score |
|:----------------------------------------|------------:|---------:|-----------:|
| topping_up_by_card                      |       0.643 |    0.450 |      0.529 |
| order_physical_card                     |       0.714 |    0.500 |      0.588 |
| supported_cards_and_currencies          |       0.568 |    0.625 |      0.595 |
| balance_not_updated_after_bank_transfer |       0.688 |    0.550 |      0.611 |
| transfer_not_received_by_recipient      |       0.614 |    0.675 |      0.643 |
| get_physical_card                       |       0.566 |    0.750 |      0.645 |
| fiat_currency_support                   |       0.630 |    0.725 |      0.674 |
| beneficiary_not_allowed                 |       0.758 |    0.625 |      0.685 |
| pending_transfer                        |       0.700 |    0.700 |      0.700 |
| transfer_fee_charged                    |       0.923 |    0.600 |      0.727 |

### Confusiones más frecuentes

| real                                    | predicha                           |   veces |
|:----------------------------------------|:-----------------------------------|--------:|
| order_physical_card                     | get_physical_card                  |      13 |
| card_acceptance                         | supported_cards_and_currencies     |      12 |
| balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient |      10 |
| beneficiary_not_allowed                 | failed_transfer                    |       9 |
| supported_cards_and_currencies          | fiat_currency_support              |       9 |
| transfer_fee_charged                    | top_up_by_bank_transfer_charge     |       9 |
| fiat_currency_support                   | exchange_via_app                   |       8 |
| card_delivery_estimate                  | card_arrival                       |       8 |
| declined_transfer                       | declined_card_payment              |       8 |
| pending_transfer                        | transfer_timing                    |       8 |
| wrong_exchange_rate_for_cash_withdrawal | cash_withdrawal_charge             |       7 |
| topping_up_by_card                      | transfer_into_account              |       6 |
