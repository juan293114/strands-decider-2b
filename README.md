# Strands Decider 2B: clasificación de intenciones con BANKING77

Experimento para medir qué tan bien clasifica **Strands Decider 2B** (AWS Strands Labs) las 77 intenciones del conjunto **BANKING77**, sin entrenamiento previo (*zero-shot*). El modelo se consume a través de **LiteLLM**.

## Resultados

Medidos el 2026-10-26 con `strands-decider-2B-hobson-v21`, sobre 770 consultas de prueba: 10 por intención.

| Método | Accuracy | Macro-F1 | Top-3 | Calibración (ECE) | Latencia mediana |
|---|---|---|---|---|---|
| TF-IDF + SVM, supervisado con 10 003 ejemplos | 0,916 | 0,915 | | | |
| Decider, solo nombres de intención | **0,843** | **0,839** | 0,962 | 0,034 | 83 ms |
| Decider, nombres en orden barajado | 0,842 | 0,840 | 0,964 | 0,041 | 84 ms |
| Decider, nombres con descripción | 0,801 | 0,798 | 0,953 | 0,205 | 212 ms |

Lo que muestran:

- **Sin ver un solo ejemplo**, el decider acierta el 84 % de 77 clases. Queda a 7 puntos de un modelo entrenado con 10 000 ejemplos.
- **La clase correcta está entre sus 3 primeras opciones el 96 % de las veces.** Sirve bien para proponer candidatas a un humano o a un segundo paso.
- **Su probabilidad es fiable.** Si solo se aceptan las predicciones con probabilidad de al menos 0,9, se cubre la mitad de las consultas con un 98,5 % de acierto.
- **El orden de las opciones no importa.** Barajarlas no cambia el resultado.
- **Añadir descripciones empeora todo.** Baja 4 puntos de accuracy, la calibración se degrada y la latencia se multiplica por 2,5.
- **Los errores se concentran en intenciones casi sinónimas**, como `card_delivery_estimate` frente a `card_arrival` u `order_physical_card` frente a `get_physical_card`.

El detalle por condición está en [results/report.md](results/report.md).

## Cómo funciona

Strands Decider no genera texto: recibe un texto (`state`) y una pregunta con opciones, y devuelve una probabilidad para cada opción. Cada consulta de BANKING77 se envía así:

```json
{
  "state": "still waiting on my new card",
  "questions": {
    "intencion": {
      "type": "choice",
      "instructions": "This is a message from a customer of a digital bank. Which customer-support intent does the message express?",
      "criteria": {"card arrival": null, "activate my card": null, "...": null}
    }
  }
}
```

En la condición `descriptions`, cada opción lleva una frase corta en lugar de `null`. Están en [data/intent_descriptions.json](data/intent_descriptions.json).

## Estructura

```
strands-decider-2b/
├── 1_download_data.py          # Paso 1: descarga BANKING77
├── 2_run_experiment.py         # Paso 2: consulta al decider y guarda respuestas
├── 3_analyze_results.py        # Paso 3: métricas y reporte
├── experiment/
│   ├── config.py               # rutas, instrucción y lectura de .env
│   ├── data.py                 # lectura de BANKING77 y armado de opciones
│   ├── decider_client.py       # cliente de LiteLLM (pydantic + SDK de OpenAI)
│   └── metrics.py              # accuracy, F1, calibración, referencia TF-IDF
├── data/
│   └── intent_descriptions.json
├── results/                    # un .jsonl por condición y report.md
└── docs/                       # guía de consumo de LiteLLM y colección de Postman
```

## Cómo reproducirlo

Requiere Python 3.10 o superior.

```bash
cd strands-decider-2b
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # y pon tu llave virtual de LiteLLM en .env
python 1_download_data.py
python 2_run_experiment.py --prueba    # muestra una petición sin enviarla
python 2_run_experiment.py             # 10 por intención, ambas condiciones
python 2_run_experiment.py --condicion names --barajar
python 3_analyze_results.py
```

> **La llave nunca se sube a git.** El archivo `.env` está en `.gitignore`.

Los resultados de esta corrida ya están en `results/`. Si relanzas el paso 2, continúa donde quedó y no repite consultas. Para empezar de cero, borra los `.jsonl`.

### Tiempo estimado

Cada consulta tarda unos 140 ms con nombres y unos 270 ms con descripciones, incluyendo la red.

| Ejecución | Consultas | Tiempo |
|---|---|---|
| 10 por intención, ambas condiciones | 1 540 | unos 5 minutos |
| Orden barajado, solo nombres | 770 | unos 2 minutos |
| Test completo, ambas condiciones (`--por-clase 0`) | 6 160 | unos 21 minutos |

## Notas

- **Una sola petición a la vez.** El servidor del decider no admite concurrencia. No lances dos ejecuciones simultáneas.
- **Ventana de contexto.** La pregunta ocupa unos 625 tokens con nombres y unos 1 450 con descripciones. El script avisa si una petición pasa de 2 800, porque por encima del límite el servidor recorta sin avisar.
- **Sin TLS.** El endpoint es HTTP. BANKING77 es público, pero no envíes datos reales de clientes hasta tener HTTPS.

## Referencias

- Strands Decider: <https://github.com/strands-labs/strands-decider>
- BANKING77: Casanueva et al., 2020, *Efficient Intent Detection with Dual Sentence Encoders*, <https://arxiv.org/abs/2003.04807>. Licencia CC BY 4.0.
