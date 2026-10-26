"""Paso 1: descarga BANKING77 (PolyAI, licencia CC BY 4.0) en la carpeta datos/.

    python 1_descargar_datos.py
"""
import urllib.request

import pandas as pd

from experimento.config import CARPETA_DATOS, URL_BANKING77

for particion in ("train", "test"):
    destino = CARPETA_DATOS / f"{particion}.csv"
    urllib.request.urlretrieve(URL_BANKING77.format(particion=particion), destino)
    df = pd.read_csv(destino)
    print(f"{particion}: {len(df)} ejemplos, {df.category.nunique()} intenciones -> {destino}")
