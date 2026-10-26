"""Configuración central: rutas, parámetros fijos y lectura del archivo .env."""
import os
from pathlib import Path

# ---- Rutas -----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DATOS = RAIZ / "datos"
CARPETA_RESULTADOS = RAIZ / "resultados"
ARCHIVO_DESCRIPCIONES = CARPETA_DATOS / "descripciones_intenciones.json"
ARCHIVO_ENV = RAIZ / ".env"

# ---- Datos -----------------------------------------------------------------------------
URL_BANKING77 = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/{particion}.csv"
)

# ---- Pregunta al modelo ------------------------------------------------------------------
# Texto que acompaña a las 77 opciones en la pregunta `choice`.
INSTRUCCION = (
    "This is a message from a customer of a digital bank. "
    "Which customer-support intent does the message express?"
)
CONDICIONES = ("nombres", "descripciones")

# ---- Límites del servidor ----------------------------------------------------------------
# La ventana del modelo es de ~3072 tokens. Por encima de este valor avisamos,
# porque el servidor recorta sin avisar y se perderían opciones.
TOKENS_ALERTA = 2800


def load_credentials() -> tuple[str, str]:
    """Lee LITELLM_BASE_URL y LITELLM_API_KEY desde el entorno o desde .env."""
    if ARCHIVO_ENV.exists():
        for linea in ARCHIVO_ENV.read_text().splitlines():
            linea = linea.strip()
            if linea and not linea.startswith("#") and "=" in linea:
                clave, valor = linea.removeprefix("export ").split("=", 1)
                os.environ.setdefault(clave.strip(), valor.strip().strip("\"'"))

    url = os.environ.get("LITELLM_BASE_URL")
    llave = os.environ.get("LITELLM_API_KEY")
    if not url or not llave:
        raise SystemExit("Faltan credenciales: copia .env.example a .env y pon tu llave virtual.")
    return url, llave
