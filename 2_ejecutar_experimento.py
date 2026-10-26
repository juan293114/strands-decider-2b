"""Paso 2: envía cada consulta de prueba al decider y guarda sus respuestas.

Cada consulta se manda como una pregunta `choice` con las 77 intenciones como
opciones. Las peticiones van una a una, porque el servidor no admite concurrencia.
Los resultados se guardan en resultados/<condicion>.jsonl; si se corta, al
relanzar continúa donde quedó.

Ejemplos:
    python 2_ejecutar_experimento.py --prueba                  # muestra una petición, sin enviar
    python 2_ejecutar_experimento.py                           # 10 por clase, ambas condiciones
    python 2_ejecutar_experimento.py --condicion nombres --barajar
    python 2_ejecutar_experimento.py --por-clase 0             # test completo (3 080)
"""
import argparse
import json
import random
import time

from experimento.cliente_decider import Choice, DeciderClient, DeciderRequest
from experimento.config import CARPETA_RESULTADOS, CONDICIONES, INSTRUCCION, TOKENS_ALERTA, load_credentials
from experimento.datos import (build_options, read_split, list_intents,
                               stratified_sample, readable_name)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ejecuta el experimento BANKING77 con Strands Decider.")
    p.add_argument("--condicion", nargs="+", choices=CONDICIONES, default=list(CONDICIONES),
                   help="cómo se presentan las opciones (por defecto: ambas)")
    p.add_argument("--por-clase", type=int, default=10,
                   help="ejemplos de prueba por intención; 0 = los 3 080 (por defecto: 10)")
    p.add_argument("--barajar", action="store_true",
                   help="baraja el orden de las 77 opciones en cada petición")
    p.add_argument("--semilla", type=int, default=13)
    p.add_argument("--prueba", action="store_true",
                   help="muestra la primera petición y termina, sin llamar al servidor")
    return p.parse_args()


def ask_decider(cliente: DeciderClient, texto: str, opciones: dict, reintentos: int = 4):
    """Envía una consulta al decider, con reintentos ante errores de red."""
    peticion = DeciderRequest(
        state=texto,
        questions={"intencion": Choice(instructions=INSTRUCCION, criteria=opciones)},
    )
    for intento in range(reintentos + 1):
        try:
            return cliente.decide(peticion)
        except Exception as error:
            if intento == reintentos:
                raise
            espera = 2 ** (intento + 1)
            print(f"  error: {error}. Reintento en {espera} s")
            time.sleep(espera)


def completed_ids(archivo) -> set[int]:
    if not archivo.exists():
        return set()
    return {json.loads(linea)["id"] for linea in archivo.open() if linea.strip()}


def to_record(fila, respuesta, latencia_total_ms: float, de_legible: dict) -> dict:
    """Convierte la respuesta del decider en una fila de resultados."""
    r = respuesta.answers["intencion"]
    probabilidades = {de_legible[k]: v for k, v in r.probabilities.items()}
    ranking = sorted(probabilidades, key=probabilidades.get, reverse=True)
    predicha = de_legible[r.choice]
    return {
        "id": int(fila.id),
        "texto": fila.texto,
        "real": fila.intencion,
        "predicha": predicha,
        "p_predicha": probabilidades[predicha],
        "p_real": probabilidades[fila.intencion],
        "posicion_real": ranking.index(fila.intencion) + 1,
        "confianza": r.confidence,
        "top5": [[i, probabilidades[i]] for i in ranking[:5]],
        "n_opciones": len(probabilidades),
        "tokens_entrada": respuesta.usage.input_tokens,
        "latencia_servidor_ms": respuesta.latency_ms,
        "latencia_total_ms": round(latencia_total_ms, 1),
        "modelo": respuesta.model,
    }


def run_condition(cliente, condicion, consultas, intenciones, args) -> None:
    nombre = condicion + ("_barajado" if args.barajar else "")
    archivo = CARPETA_RESULTADOS / f"{nombre}.jsonl"
    hechos = completed_ids(archivo)
    pendientes = consultas[~consultas.id.isin(hechos)]
    print(f"[{nombre}] {len(hechos)} hechas, {len(pendientes)} pendientes -> {archivo.name}")

    de_legible = {readable_name(i): i for i in intenciones}
    inicio = time.perf_counter()
    with archivo.open("a") as salida:
        for n, fila in enumerate(pendientes.itertuples(), start=1):
            # Semilla por ejemplo: el orden barajado se repite igual aunque se reanude.
            barajador = random.Random(args.semilla * 100_003 + fila.id) if args.barajar else None
            opciones = build_options(condicion, intenciones, barajador)

            t0 = time.perf_counter()
            respuesta = ask_decider(cliente, fila.texto, opciones)
            registro = to_record(fila, respuesta, (time.perf_counter() - t0) * 1000, de_legible)

            if registro["tokens_entrada"] > TOKENS_ALERTA:
                print(f"  aviso: {registro['tokens_entrada']} tokens; el servidor podría recortar")
            salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            salida.flush()

            if n % 50 == 0 or n == len(pendientes):
                por_peticion = (time.perf_counter() - inicio) / n
                restante = por_peticion * (len(pendientes) - n) / 60
                print(f"  {n}/{len(pendientes)}  {por_peticion * 1000:.0f} ms/petición  "
                      f"faltan {restante:.1f} min")


def main() -> None:
    args = parse_arguments()
    intenciones = list_intents()
    consultas = stratified_sample(read_split("test"), args.por_clase, args.semilla)
    print(f"{len(consultas)} consultas por condición, {len(intenciones)} intenciones")

    if args.prueba:
        opciones = build_options(args.condicion[0], intenciones)
        peticion = DeciderRequest(state=consultas.texto[0], questions={
            "intencion": Choice(instructions=INSTRUCCION, criteria=opciones)})
        print(json.dumps(peticion.model_dump(), indent=1, ensure_ascii=False)[:1200], "...")
        return

    cliente = DeciderClient(*load_credentials())
    # Calentamiento: la primera petición tras un despliegue puede tardar decenas de segundos.
    ask_decider(cliente, consultas.texto[0], build_options("nombres", intenciones))

    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    for condicion in args.condicion:
        run_condition(cliente, condicion, consultas, intenciones, args)


if __name__ == "__main__":
    main()
