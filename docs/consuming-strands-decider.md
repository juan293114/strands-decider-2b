# Consumir Strands Decider vía LiteLLM

Base URL: `http://loadbalancer-litellm-ext-1322139303.us-east-1.elb.amazonaws.com` (HTTP, sin TLS por ahora)
Modelo: `strands-decider` · API compatible con OpenAI `/v1/chat/completions` · Auth: `Authorization: Bearer <llave virtual>`

## Regla clave
El decider **no es un LLM**. El contenido del mensaje `user` debe ser un **string JSON** con `state` y `questions`.
La respuesta del decider llega como **string JSON** en `choices[0].message.content` (hay que hacer `json.loads`).
El Playground de la UI de LiteLLM con texto libre devuelve 400: es lo esperado.

## Preguntas (`questions`: nombre -> pregunta)
| type | campos | respuesta |
|---|---|---|
| `noul` | `instructions` | `noul`: probabilidad (0-1) |
| `choice` | `instructions`, `criteria`: {opción: descripción o null}, 2-255 | `choice`, `probabilities`, `confidence` |
| `score` | `instructions`, `criteria`: lista ascendente, 2-10 niveles | `score` (esperanza; 0 = primer nivel), `legend`, `probabilities`, `confidence` |

## Respuesta (ejemplo real)
```json
{"model":"strands-decider-2B-hobson-v21",
 "answers":{"is_urgent":{"type":"noul","noul":0.8755},
            "team":{"type":"choice","choice":"billing","probabilities":{"billing":0.8897,"sales":0.0559,"retail":0.0544},"confidence":0.8345},
            "frustration":{"type":"score","score":1.0725,"legend":{"0":"calm","1":"frustrated","2":"depressed"},"probabilities":{"0":0.1401,"1":0.6474,"2":0.2125},"confidence":0.6006}},
 "usage":{"input_tokens":196,"output_tokens":3},"latency_ms":156.18}
```

## Python
`decider_client.py` (esta carpeta): modelos pydantic de petición y respuesta + `DeciderClient`. Requiere `openai` y `pydantic`.

## curl
```bash
curl -s $BASE/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{
 "model":"strands-decider",
 "messages":[{"role":"user","content":"{\"state\":\"Help! My payouts are failing\",\"questions\":{\"is_urgent\":{\"type\":\"noul\",\"instructions\":\"Does this convey urgency?\"}}}"}]}'
```

## Postman
Importar `litellm-strands-decider.postman_collection.json` y rellenar la variable `api_key`.

La carpeta **Clasificación – agentes de voz** trae 29 ejemplos en español de Colombia, en cinco subcarpetas: intención, control del turno, fraude, salientes y calidad post-llamada. Cada petición incluye tests con la respuesta esperada y una tabla en la pestaña *Visualize*; la descripción de cada carpeta resume las reglas medidas. Ejecutar las carpetas en serie, nunca en paralelo.

## Notas operativas
- **Una sola petición en vuelo**: el servidor del decider no es seguro con peticiones concurrentes (puede dar respuestas erróneas o quedarse sin memoria de GPU).
- **`state` ≤ ~1.500 tokens**: si no cabe, el servidor corta por la derecha sin avisar y se pierden los turnos más recientes.
- Pedir una **llave virtual** (`POST /key/generate`) al administrador; no repartir la master key.
- Timeout del cliente >= 120 s (la 1a petición tras desplegar el endpoint tarda ~42 s; en régimen ~160 ms).
- Sin streaming ni parámetros de generación (temperature, etc.): se ignoran.
- Sin TLS: no enviar datos sensibles reales hasta tener HTTPS.
