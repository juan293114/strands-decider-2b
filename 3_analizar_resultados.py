"""Paso 3: calcula las métricas y escribe resultados/reporte.md.

    python 3_analizar_resultados.py
"""
import pandas as pd

from experimento import metricas
from experimento.config import CARPETA_RESULTADOS
from experimento.datos import leer_particion


def tabla(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    corridas = {f.stem: pd.read_json(f, lines=True)
                for f in sorted(CARPETA_RESULTADOS.glob("*.jsonl"))}
    if not corridas:
        raise SystemExit("No hay resultados. Ejecuta primero: python 2_ejecutar_experimento.py")

    # Comparamos solo los ejemplos presentes en todas las condiciones.
    comunes = set.intersection(*(set(df.id) for df in corridas.values()))
    corridas = {k: df[df.id.isin(comunes)] for k, df in corridas.items()}

    test = leer_particion("test")
    referencia = metricas.referencia_supervisada(leer_particion("train"),
                                                 test[test.id.isin(comunes)])
    filas = [{"condicion": "TF-IDF + SVM (supervisado)", **referencia}]
    filas += [{"condicion": f"Decider: {k}", **metricas.resumen(df)} for k, df in corridas.items()]
    resumen = pd.DataFrame(filas)

    modelos = sorted({m for df in corridas.values() for m in df.modelo.unique()})
    lineas = [
        "# Resultados: BANKING77 con Strands Decider", "",
        f"Modelo servido: {', '.join(modelos)}. Ejemplos de prueba: {len(comunes)}.", "",
        "La latencia es la que reporta el servidor del decider, sin red ni LiteLLM.", "",
        "## Resumen", "", tabla(resumen), "",
    ]
    for nombre, df in corridas.items():
        lineas += [
            f"## Condición: {nombre}", "",
            f"Tokens de entrada: mediana {df.tokens_entrada.median():.0f}, "
            f"máximo {df.tokens_entrada.max()}.", "",
            "### Exactitud según umbral de probabilidad", "", tabla(metricas.tabla_umbrales(df)), "",
            "### Diez intenciones con peor F1", "", tabla(metricas.peores_clases(df)), "",
            "### Confusiones más frecuentes", "", tabla(metricas.confusiones(df)), "",
        ]

    reporte = CARPETA_RESULTADOS / "reporte.md"
    reporte.write_text("\n".join(lineas))
    print(tabla(resumen))
    print(f"\nReporte completo en {reporte}")


if __name__ == "__main__":
    main()
