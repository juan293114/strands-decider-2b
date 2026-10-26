"""Paso 4 (opcional): ajusta BERT con BANKING77 para tener una referencia propia.

En lugar de copiar la cifra del artículo original, entrenamos BERT nosotros con
los 10 003 ejemplos de entrenamiento y lo evaluamos con las mismas 3 080
consultas de prueba que el decider. Igual que el artículo, no se reserva una
parte para validación: se entrena un número fijo de épocas.

Requiere las dependencias extra:  pip install -r requirements-bert.txt
Una GPU lo hace en minutos; en CPU tarda horas.

Ejemplos:
    python 4_train_bert.py                                        # BERT-Base
    python 4_train_bert.py --modelo google-bert/bert-large-uncased --lr 2e-5
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from experiment.config import CARPETA_RESULTADOS
from experiment.data import list_intents, read_split

CARPETA_BERT = CARPETA_RESULTADOS / "bert"


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ajusta BERT con BANKING77 y evalúa en el test completo.")
    p.add_argument("--modelo", default="google-bert/bert-base-uncased",
                   help="modelo de Hugging Face o carpeta local (por defecto: BERT-Base)")
    p.add_argument("--epocas", type=int, default=5)
    p.add_argument("--lr", type=float, default=4e-5, help="tasa de aprendizaje")
    p.add_argument("--lote", type=int, default=32, help="ejemplos por paso de entrenamiento")
    p.add_argument("--max-tokens", type=int, default=64, help="longitud máxima de cada consulta")
    p.add_argument("--semilla", type=int, default=13)
    p.add_argument("--limite-train", type=int, default=0,
                   help="usa solo N ejemplos de entrenamiento (para pruebas rápidas)")
    return p.parse_args()


def set_seed(semilla: int) -> None:
    """Fija las semillas para que el entrenamiento sea reproducible."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batches(textos, etiquetas, tokenizer, max_tokens, lote, barajar):
    """Agrupa los ejemplos en lotes ya tokenizados."""
    ejemplos = list(zip(textos, etiquetas))

    def collate(grupo):
        t, e = zip(*grupo)
        lote_tok = tokenizer(list(t), padding=True, truncation=True,
                             max_length=max_tokens, return_tensors="pt")
        lote_tok["labels"] = torch.tensor(e)
        return lote_tok

    return DataLoader(ejemplos, batch_size=lote, shuffle=barajar, collate_fn=collate)


def train(modelo, lotes, epocas, lr, dispositivo) -> None:
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    # En GPU usamos precisión mixta (float16): BERT-Large entrena varias veces más rápido.
    # En CPU queda desactivada y el código hace exactamente lo mismo en float32.
    usar_amp = dispositivo.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)
    pasos = epocas * len(lotes)
    # La tasa de aprendizaje sube durante el primer 10 % y luego baja linealmente a 0.
    programador = get_linear_schedule_with_warmup(optimizador, int(0.1 * pasos), pasos)

    modelo.train()
    inicio = time.perf_counter()
    for epoca in range(1, epocas + 1):
        perdida_total = 0.0
        for lote in lotes:
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=usar_amp):
                perdida = modelo(**lote).loss
            escalador.scale(perdida).backward()
            escalador.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escalador.step(optimizador)
            escalador.update()
            programador.step()
            optimizador.zero_grad()
            perdida_total += perdida.item()
        minutos = (time.perf_counter() - inicio) / 60
        print(f"  época {epoca}/{epocas}  pérdida {perdida_total / len(lotes):.4f}  {minutos:.1f} min")


@torch.no_grad()
def predict(modelo, lotes, dispositivo) -> np.ndarray:
    """Devuelve la matriz de probabilidades (consultas x intenciones)."""
    modelo.eval()
    probabilidades = []
    for lote in lotes:
        lote = {k: v.to(dispositivo) for k, v in lote.items() if k != "labels"}
        probabilidades.append(torch.softmax(modelo(**lote).logits, dim=-1).cpu().numpy())
    return np.concatenate(probabilidades)


def to_records(test, probabilidades, intenciones, nombre_modelo) -> list[dict]:
    """Mismos campos que los resultados del decider, para analizarlos igual."""
    registros = []
    for fila, p in zip(test.itertuples(), probabilidades):
        ranking = np.argsort(-p)
        real = intenciones.index(fila.intencion)
        registros.append({
            "id": int(fila.id),
            "texto": fila.texto,
            "real": fila.intencion,
            "predicha": intenciones[ranking[0]],
            "p_predicha": float(p[ranking[0]]),
            "p_real": float(p[real]),
            "posicion_real": int(np.where(ranking == real)[0][0]) + 1,
            "top5": [[intenciones[i], float(p[i])] for i in ranking[:5]],
            "modelo": nombre_modelo,
        })
    return registros


def main() -> None:
    args = parse_arguments()
    set_seed(args.semilla)
    dispositivo = pick_device()

    intenciones = list_intents()
    indice = {intencion: i for i, intencion in enumerate(intenciones)}
    train_df, test_df = read_split("train"), read_split("test")
    if args.limite_train:
        train_df = train_df.sample(n=args.limite_train, random_state=args.semilla)
    print(f"Modelo {args.modelo} en {dispositivo}: "
          f"{len(train_df)} ejemplos de entrenamiento, {len(test_df)} de prueba")

    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        args.modelo, num_labels=len(intenciones)).to(dispositivo)

    lotes_train = make_batches(train_df.texto, train_df.intencion.map(indice),
                               tokenizer, args.max_tokens, args.lote, barajar=True)
    lotes_test = make_batches(test_df.texto, test_df.intencion.map(indice),
                              tokenizer, args.max_tokens, 128, barajar=False)

    train(modelo, lotes_train, args.epocas, args.lr, dispositivo)
    probabilidades = predict(modelo, lotes_test, dispositivo)

    nombre = args.modelo.rstrip("/").split("/")[-1]
    registros = to_records(test_df, probabilidades, intenciones, nombre)
    accuracy = np.mean([r["real"] == r["predicha"] for r in registros])
    print(f"Accuracy en las {len(registros)} consultas de prueba: {accuracy:.4f}")

    CARPETA_BERT.mkdir(parents=True, exist_ok=True)
    salida = CARPETA_BERT / f"{nombre}.jsonl"
    with salida.open("w") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    config = vars(args) | {"dispositivo": str(dispositivo), "accuracy": accuracy}
    (CARPETA_BERT / f"{nombre}.config.json").write_text(json.dumps(config, indent=1))
    print(f"Predicciones guardadas en {salida}")


if __name__ == "__main__":
    main()
