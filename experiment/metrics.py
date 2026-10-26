"""Métricas de clasificación y modelo supervisado de referencia."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC


def expected_calibration_error(confianza: np.ndarray, acierto: np.ndarray, intervalos: int = 15) -> float:
    """ECE: diferencia media entre la probabilidad que da el modelo y su acierto real.

    0 significa calibración perfecta: cuando dice 80 %, acierta el 80 % de las veces.
    """
    bordes = np.linspace(0, 1, intervalos + 1)
    total = 0.0
    for bajo, alto in zip(bordes[:-1], bordes[1:]):
        en_intervalo = (confianza > bajo) & (confianza <= alto)
        if en_intervalo.any():
            hueco = abs(acierto[en_intervalo].mean() - confianza[en_intervalo].mean())
            total += en_intervalo.mean() * hueco
    return float(total)


def summarize(df: pd.DataFrame) -> dict:
    """Métricas principales de una condición (un DataFrame de resultados)."""
    acierto = (df.real == df.predicha).to_numpy(float)
    return {
        "accuracy": acierto.mean(),
        "macro_f1": f1_score(df.real, df.predicha, average="macro"),
        "top3": (df.posicion_real <= 3).mean(),
        "top5": (df.posicion_real <= 5).mean(),
        "ECE": expected_calibration_error(df.p_predicha.to_numpy(), acierto),
        "latencia_p50_ms": df.latencia_servidor_ms.median(),
        "latencia_p95_ms": df.latencia_servidor_ms.quantile(0.95),
    }


def threshold_table(df: pd.DataFrame, umbrales=(0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """Si solo aceptamos predicciones con probabilidad >= umbral:
    qué fracción se acepta (cobertura) y qué tan exactas son."""
    filas = []
    for u in umbrales:
        aceptadas = df.p_predicha >= u
        exactitud = (df.real == df.predicha)[aceptadas].mean() if aceptadas.any() else np.nan
        filas.append({"umbral": u, "cobertura": aceptadas.mean(), "accuracy_aceptadas": exactitud})
    return pd.DataFrame(filas)


def worst_classes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    reporte = pd.DataFrame(classification_report(
        df.real, df.predicha, output_dict=True, zero_division=0)).T
    reporte = reporte.drop(index=["accuracy", "macro avg", "weighted avg"])
    reporte = reporte.sort_values("f1-score").head(n)
    return reporte.reset_index(names="intencion")[["intencion", "precision", "recall", "f1-score"]]


def top_confusions(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    errores = df[df.real != df.predicha]
    return (errores.groupby(["real", "predicha"]).size()
            .sort_values(ascending=False).head(n).reset_index(name="veces"))


def supervised_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """TF-IDF de palabras y caracteres + SVM lineal, entrenado con los 10 003 ejemplos."""
    modelo = make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        ),
        LinearSVC(C=0.5),
    )
    modelo.fit(train.texto, train.intencion)
    predicha = modelo.predict(test.texto)
    return {"accuracy": accuracy_score(test.intencion, predicha),
            "macro_f1": f1_score(test.intencion, predicha, average="macro")}
