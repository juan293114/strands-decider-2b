# Strands Decider 2B: clasificación de intenciones con BANKING77

Experimento para medir qué tan bien clasifica **Strands Decider 2B** (AWS Strands Labs) las 77 intenciones del conjunto **BANKING77**, sin entrenamiento previo (*zero-shot*). El modelo se consume a través de **LiteLLM**.

## Resultados

Medidos el 2026-10-26 con `strands-decider-2B-hobson-v21` sobre el **test completo de BANKING77: 3 080 consultas**, 40 por intención. Se hicieron 9 240 peticiones al decider en total, sin ningún error.

| Método | Datos de entrenamiento | Accuracy | Macro-F1 | Top-3 | Calibración (ECE) | Latencia mediana |
|---|---|---|---|---|---|---|
| BERT ajustado, artículo original¹ | 10 003 ejemplos | 0,937 | | | | |
| USE + ConveRT, artículo original¹ | 10 003 ejemplos | 0,934 | | | | |
| TF-IDF + SVM, referencia propia² | 10 003 ejemplos | 0,913 | 0,913 | | | |
| **Decider, solo nombres de intención** | **ninguno** | **0,844** | **0,842** | **0,969** | **0,022** | **83 ms** |
| Decider, nombres en orden barajado | ninguno | 0,844 | 0,844 | 0,969 | 0,019 | 84 ms |
| Decider, nombres con descripción | ninguno | 0,811 | 0,810 | 0,955 | 0,209 | 213 ms |

¹ Casanueva et al., 2020. Cifras tomadas de la tabla del artículo reproducida en un artículo de NAACL 2021; conviene confirmarlas en el PDF original.
² Calculada con el código de este repositorio, en `experiment/metrics.py`.

La latencia es la que reporta el servidor del decider. Con la red y LiteLLM, cada consulta tarda unos 143 ms con nombres y unos 272 ms con descripciones.

Lo que muestran:

- **Sin ver un solo ejemplo**, el decider acierta el 84 % de 77 clases. Queda a 7 puntos de un TF-IDF supervisado y a 9 puntos del mejor modelo del artículo.
- **La clase correcta está entre sus 3 primeras opciones el 97 % de las veces.** Sirve bien para proponer candidatas a un humano o a un segundo paso.
- **Su probabilidad es fiable.** Con nombres, el error de calibración es de solo 0,022. Si solo se aceptan las predicciones con probabilidad de al menos 0,9, se cubre el 52 % de las consultas con un 97,5 % de acierto.
- **El orden de las opciones no cambia la exactitud global.** Aun así, el 6,7 % de las predicciones individuales cambian al barajar.
- **Añadir descripciones empeora todo.** Baja 3 puntos de accuracy, la calibración se degrada y la latencia se multiplica por 2,5.
- **Los errores se concentran en intenciones casi sinónimas.** Las confusiones más frecuentes son `order_physical_card` con `get_physical_card`, y `card_delivery_estimate` con `card_arrival`.

### Exactitud según la probabilidad mínima exigida

| Probabilidad mínima | Consultas aceptadas, solo nombres | Accuracy en las aceptadas |
|---|---|---|
| Sin umbral | 100 % | 0,844 |
| 0,8 | 66 % | 0,947 |
| 0,9 | 52 % | 0,975 |

El detalle completo por condición está en [results/report.md](results/report.md).

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

Se comparan tres condiciones:

| Condición | Cómo se presentan las 77 opciones |
|---|---|
| `names` | Solo el nombre legible de cada intención, por ejemplo `card arrival` |
| `descriptions` | El nombre más una frase corta que describe la intención |
| `names_shuffled` | Solo nombres, con el orden de las opciones barajado en cada consulta |

## Descripción de cada archivo

### Scripts principales

Se ejecutan en orden desde la raíz del repositorio.

| Archivo | Qué hace |
|---|---|
| `1_download_data.py` | Descarga los CSV de entrenamiento y prueba de BANKING77 en la carpeta `data/`. |
| `2_run_experiment.py` | Envía cada consulta de prueba al decider vía LiteLLM, una a una, y guarda cada respuesta en `results/<condición>.jsonl`. Si se interrumpe, al relanzarlo continúa donde quedó. |
| `3_analyze_results.py` | Lee los resultados, entrena la referencia TF-IDF, calcula todas las métricas y escribe `results/report.md`. |

### Paquete `experiment/`

| Archivo | Qué contiene |
|---|---|
| `__init__.py` | Marca la carpeta como paquete de Python. |
| `config.py` | Rutas de carpetas, la instrucción que acompaña a las opciones, el umbral de alerta de tokens y la función `load_credentials`, que lee la URL y la llave desde `.env`. |
| `data.py` | Lectura de BANKING77 con `read_split`, muestreo por intención con `stratified_sample`, y armado de las opciones del decider con `build_options`. |
| `decider_client.py` | Cliente para LiteLLM: modelos pydantic de la petición y la respuesta, y la clase `DeciderClient`, que usa el SDK de OpenAI. |
| `metrics.py` | Accuracy, macro-F1, top-k, error de calibración, tabla por umbral, peores intenciones, confusiones frecuentes y la referencia TF-IDF + SVM. |

### Datos

| Archivo | Qué contiene |
|---|---|
| `data/intent_descriptions.json` | Una frase corta en inglés por cada una de las 77 intenciones. La usa la condición `descriptions`. |
| `data/train.csv`, `data/test.csv` | BANKING77. No se versionan: los descarga el paso 1. |

### Resultados

| Archivo | Qué contiene |
|---|---|
| `results/names.jsonl` | Una línea por consulta de la condición `names`, con las 3 080 consultas. |
| `results/descriptions.jsonl` | Igual, para la condición `descriptions`. |
| `results/names_shuffled.jsonl` | Igual, para la condición `names_shuffled`. |
| `results/report.md` | Reporte generado por el paso 3, con todas las tablas de métricas. |

Cada línea de los `.jsonl` tiene estos campos:

| Campo | Significado |
|---|---|
| `id`, `texto`, `real` | Posición en `test.csv`, texto de la consulta e intención correcta |
| `predicha`, `p_predicha` | Intención elegida por el decider y su probabilidad |
| `p_real`, `posicion_real` | Probabilidad que dio a la intención correcta y su puesto en el ranking |
| `confianza` | Confianza normalizada que devuelve el decider |
| `top5` | Las cinco intenciones más probables con su probabilidad |
| `n_opciones`, `tokens_entrada` | Número de opciones enviadas y tokens que procesó el servidor |
| `latencia_servidor_ms`, `latencia_total_ms` | Tiempo del modelo y tiempo total visto por el cliente |
| `modelo` | Versión del decider que respondió |

### Documentación y configuración

| Archivo | Qué contiene |
|---|---|
| `docs/consuming-strands-decider.md` | Guía para consumir el decider vía LiteLLM: formato de peticiones, respuestas y notas operativas. |
| `docs/litellm-strands-decider.postman_collection.json` | Colección de Postman con ejemplos de peticiones. Hay que rellenar la variable `api_key`. |
| `.env.example` | Plantilla del archivo `.env` con la URL de LiteLLM y un espacio para la llave. |
| `.gitignore` | Evita subir `.env`, los CSV descargados y archivos temporales. |
| `requirements.txt` | Dependencias de Python. |

## Cómo reproducirlo

Requiere Python 3.10 o superior.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # y pon tu llave virtual de LiteLLM en .env
python 1_download_data.py
python 2_run_experiment.py --prueba                          # muestra una petición sin enviarla
python 2_run_experiment.py --por-clase 1                     # lote pequeño de validación
python 2_run_experiment.py --por-clase 0                     # test completo, ambas condiciones
python 2_run_experiment.py --por-clase 0 --condicion names --barajar
python 3_analyze_results.py
```

> **La llave nunca se sube a git.** El archivo `.env` está en `.gitignore`.

Los resultados del test completo ya están en `results/`. Si relanzas el paso 2, continúa donde quedó y no repite consultas. Para empezar de cero, borra los `.jsonl`.

### Tiempo estimado

| Ejecución | Consultas | Tiempo |
|---|---|---|
| Lote de validación, `--por-clase 1`, tres condiciones | 231 | menos de 1 minuto |
| 10 por intención, dos condiciones | 1 540 | unos 5 minutos |
| Test completo, dos condiciones | 6 160 | unos 21 minutos |
| Test completo, orden barajado | 3 080 | unos 7 minutos |

## Notas

- **Una sola petición a la vez.** El servidor del decider no admite concurrencia. No lances dos ejecuciones simultáneas.
- **Ventana de contexto.** La pregunta ocupa como máximo unos 690 tokens con nombres y unos 1 510 con descripciones. El script avisa si una petición pasa de 2 800, porque por encima del límite el servidor recorta sin avisar.
- **Sin TLS.** El endpoint es HTTP. BANKING77 es público, pero no envíes datos reales de clientes hasta tener HTTPS.

## Referencias

- Strands Decider: <https://github.com/strands-labs/strands-decider>
- BANKING77: Casanueva et al., 2020, *Efficient Intent Detection with Dual Sentence Encoders*, <https://aclanthology.org/2020.nlp4convai-1.5.pdf>. Licencia CC BY 4.0.
- Artículo de NAACL 2021 que reproduce la tabla de BANKING77: <https://aclanthology.org/2021.naacl-main.237.pdf>
