"""Paso 3: calcula las métricas y escribe results/report.md.

    python 3_analyze_results.py
"""
import pandas as pd

from experiment import metrics
from experiment.config import CARPETA_RESULTADOS

# Modelos opcionales que se comparan con el decider si existen sus resultados:
# BERT (paso 4) y el decider ajustado (paso 5 y notebook de Colab).
MODELOS_EXTRA = {
    "bert": "BERT ajustado",
    "decider_finetuned": "Decider ajustado",
}
from experiment.data import read_split


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    corridas = {f.stem: pd.read_json(f, lines=True)
                for f in sorted(CARPETA_RESULTADOS.glob("*.jsonl"))}
    if not corridas:
        raise SystemExit("No hay resultados. Ejecuta primero: python 2_run_experiment.py")

    # Comparamos solo los ejemplos presentes en todas las condiciones.
    comunes = set.intersection(*(set(df.id) for df in corridas.values()))
    corridas = {k: df[df.id.isin(comunes)] for k, df in corridas.items()}

    test = read_split("test")
    referencia = metrics.supervised_baseline(read_split("train"),
                                                 test[test.id.isin(comunes)])
    filas = [{"condicion": "TF-IDF + SVM (supervisado)", **referencia}]
    # Los modelos opcionales entran en la tabla si cubren las mismas consultas.
    for carpeta, etiqueta in MODELOS_EXTRA.items():
        for archivo in sorted((CARPETA_RESULTADOS / carpeta).glob("*.jsonl")):
            extra = pd.read_json(archivo, lines=True)
            extra = extra[extra.id.isin(comunes)]
            if len(extra) == len(comunes):
                filas.append({"condicion": f"{etiqueta}: {archivo.stem}", **metrics.summarize(extra)})
            else:
                print(f"aviso: {carpeta}/{archivo.name} no cubre las mismas consultas; se omite")
    filas += [{"condicion": f"Decider: {k}", **metrics.summarize(df)} for k, df in corridas.items()]
    resumen = pd.DataFrame(filas)

    modelos = sorted({m for df in corridas.values() for m in df.modelo.unique()})
    lineas = [
        "# Resultados: BANKING77 con Strands Decider", "",
        f"Modelo servido: {', '.join(modelos)}. Ejemplos de prueba: {len(comunes)}.", "",
        "La latencia es la que reporta el servidor del decider, sin red ni LiteLLM.", "",
        "## Resumen", "", markdown_table(resumen), "",
    ]
    for nombre, df in corridas.items():
        lineas += [
            f"## Condición: {nombre}", "",
            f"Tokens de entrada: mediana {df.tokens_entrada.median():.0f}, "
            f"máximo {df.tokens_entrada.max()}.", "",
            "### Exactitud según umbral de probabilidad", "", markdown_table(metrics.threshold_table(df)), "",
            "### Diez intenciones con peor F1", "", markdown_table(metrics.worst_classes(df)), "",
            "### Confusiones más frecuentes", "", markdown_table(metrics.top_confusions(df)), "",
        ]

    reporte = CARPETA_RESULTADOS / "report.md"
    reporte.write_text("\n".join(lineas))
    print(markdown_table(resumen))
    print(f"\nReporte completo en {reporte}")


if __name__ == "__main__":
    main()
