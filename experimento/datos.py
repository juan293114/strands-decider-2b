"""Lectura de BANKING77 y construcción de las opciones que ve el modelo."""
import json
import random

import pandas as pd

from .config import ARCHIVO_DESCRIPCIONES, CARPETA_DATOS


def leer_particion(particion: str) -> pd.DataFrame:
    """Devuelve 'train' o 'test' con columnas: id, texto, intencion."""
    ruta = CARPETA_DATOS / f"{particion}.csv"
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. Ejecuta primero: python 1_descargar_datos.py")
    df = pd.read_csv(ruta).rename(columns={"text": "texto", "category": "intencion"})
    df.insert(0, "id", range(len(df)))
    return df


def muestra_estratificada(df: pd.DataFrame, por_clase: int, semilla: int) -> pd.DataFrame:
    """Toma `por_clase` ejemplos de cada intención. Con 0 devuelve todo."""
    if por_clase > 0:
        df = df.groupby("intencion").sample(n=por_clase, random_state=semilla)
    return df.sort_values("id").reset_index(drop=True)


def lista_intenciones() -> list[str]:
    """Las 77 intenciones, en orden alfabético."""
    return sorted(leer_particion("train").intencion.unique())


def nombre_legible(intencion: str) -> str:
    """'card_arrival' -> 'card arrival'. Es el texto de la opción que ve el modelo."""
    return intencion.strip("?").lower().replace("_", " ")


def construir_opciones(condicion: str, intenciones: list[str],
                       barajar_con: random.Random | None = None) -> dict[str, str | None]:
    """Arma el diccionario `criteria` de la pregunta choice.

    - "nombres":       {"card arrival": None, ...}
    - "descripciones": {"card arrival": "an ordered card has not arrived yet", ...}
    Si se pasa `barajar_con`, el orden de las opciones se baraja.
    """
    orden = list(intenciones)
    if barajar_con is not None:
        barajar_con.shuffle(orden)

    if condicion == "nombres":
        return {nombre_legible(i): None for i in orden}
    if condicion == "descripciones":
        descripciones = json.loads(ARCHIVO_DESCRIPCIONES.read_text())
        return {nombre_legible(i): descripciones[i] for i in orden}
    raise ValueError(f"Condición desconocida: {condicion}")
