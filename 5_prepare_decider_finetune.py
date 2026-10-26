"""Paso 5 (opcional): prepara el ajuste fino de Strands Decider con BANKING77.

Convierte el conjunto de entrenamiento al formato de filas del repositorio oficial
de Strands Decider y escribe la configuración de entrenamiento. El entrenamiento
en sí lo hace el código oficial (`python -m training.train`) en una GPU de 24 GB o
más; ver notebooks/colab_strands_decider_finetune.ipynb.

Archivos que escribe en work/decider/:
    train.jsonl   filas de entrenamiento (todas las del train menos las de calibración)
    calib.jsonl   filas reservadas para calibrar la confianza del modelo ajustado
    train.yaml    configuración para `python -m training.train`

Las filas usan exactamente la misma instrucción y las mismas opciones que
2_run_experiment.py envía al servir (condición `names`), para que el modelo se
entrene con lo mismo que verá después.

    python 5_prepare_decider_finetune.py
"""
import argparse
import json
import math

from experiment.config import INSTRUCCION, RAIZ
from experiment.data import build_options, list_intents, read_split, readable_name

CARPETA_DECIDER = RAIZ / "work" / "decider"

# Otras formas de hacer la misma pregunta: el entrenamiento elige una al azar en
# cada época, para que el modelo aprenda la tarea y no una redacción concreta.
VARIANTES = [
    "Which intent best describes what this bank customer is asking for?",
    "Classify the customer's message into one of the banking support intents.",
    "What does the customer of this digital bank want? Pick the matching intent.",
]

# Campos que fija el checkpoint publicado y que la configuración NO debe contener;
# el entrenamiento oficial se detiene si aparece alguno.
CAMPOS_PROHIBIDOS = {"base_model", "num_slots", "head_type", "pointer_dim", "head_hidden",
                     "use_lora", "lora_r", "lora_alpha", "lora_targets", "force_bos"}


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepara el ajuste fino de Strands Decider.")
    p.add_argument("--init-from", default="StrandsAgents/strands-decider-2B-hobson-v21",
                   help="checkpoint publicado del que se parte (por defecto: el mismo que sirve LiteLLM)")
    p.add_argument("--calib-por-clase", type=int, default=20,
                   help="ejemplos de entrenamiento por intención reservados para calibrar")
    p.add_argument("--epocas", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--checkpoint", default=str(CARPETA_DECIDER / "ckpt"),
                   help="carpeta donde el entrenamiento guardará el modelo ajustado")
    return p.parse_args()


def to_row(texto: str, intencion: str, opciones: list[list]) -> dict:
    """Una fila del formato oficial: estado, pregunta, opciones e índice correcto."""
    nombres = [nombre for nombre, _ in opciones]
    return {
        "kind": "choice",
        "state": texto,
        "instructions": INSTRUCCION,
        "options": opciones,
        "label": nombres.index(readable_name(intencion)),
        "task": "banking77",
        "instruction_variants": VARIANTES,
    }


def write_jsonl(ruta, filas) -> None:
    with ruta.open("w") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def write_config(ruta, args, n_train: int) -> int:
    """Escribe train.yaml y devuelve el número estimado de pasos de entrenamiento."""
    filas_por_paso = 8 * 4  # micro_batch_size x grad_accum
    pasos = math.ceil(n_train * 0.97 / filas_por_paso) * args.epocas
    config = {
        "init_from": args.init_from,
        "train_files": [str(CARPETA_DECIDER / "train.jsonl")],
        "val_fraction": 0.03,
        "val_split_seed": 0,
        # Sin término KL: el anclaje al modelo original solo actúa en preguntas con
        # hasta num_slots opciones, y aquí todas tienen 77.
        "kl_frozen_weight": 0.0,
        "group_by_length": True,
        "epochs": args.epocas,
        "micro_batch_size": 8,
        "grad_accum": 4,
        "lr": args.lr,
        "head_lr": args.lr,  # la cabeza ya viene entrenada
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "shuffle_options": True,  # baraja las 77 opciones en cada época
        "head_dropout": 0.05,
        "seed": args.semilla,
        "output_dir": args.checkpoint,
        "log_every": 20,
        "eval_every": max(20, pasos // 5),
    }
    assert not CAMPOS_PROHIBIDOS & config.keys()
    # JSON es YAML válido, así que no hace falta otra dependencia.
    lineas = [f"{clave}: {json.dumps(valor)}" for clave, valor in config.items()]
    ruta.write_text("\n".join(lineas) + "\n")
    return pasos


def main() -> None:
    args = parse_arguments()
    intenciones = list_intents()
    opciones = [[nombre, desc] for nombre, desc in build_options("names", intenciones).items()]

    train = read_split("train")
    calib = train.groupby("intencion").sample(n=args.calib_por_clase, random_state=13)
    train = train.drop(calib.index)

    CARPETA_DECIDER.mkdir(parents=True, exist_ok=True)
    write_jsonl(CARPETA_DECIDER / "train.jsonl",
                (to_row(f.texto, f.intencion, opciones) for f in train.itertuples()))
    write_jsonl(CARPETA_DECIDER / "calib.jsonl",
                (to_row(f.texto, f.intencion, opciones) for f in calib.itertuples()))
    pasos = write_config(CARPETA_DECIDER / "train.yaml", args, len(train))

    print(f"train.jsonl: {len(train)} filas | calib.jsonl: {len(calib)} filas | "
          f"{len(opciones)} opciones por fila")
    print(f"train.yaml: parte de {args.init_from}, {args.epocas} épocas, unos {pasos} pasos")
    print(f"Archivos en {CARPETA_DECIDER}")


if __name__ == "__main__":
    main()
