# explorar_zip_s3_filtrado.py
import boto3
import zipfile
import io
import logging
import sys
import csv
from datetime import datetime

# ── Logging ──────────────────────────────────────────────
log_filename = f"explorar_zip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,  # menos ruido
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────
BUCKET = "rayo-scout-data"
KEY    = "europa_temp.zip"
REGION = "eu-west-1"

# ── Stream ZIP desde S3 ──────────────────────────────────
class S3ZipStream:
    def __init__(self, s3_client, bucket, key):
        self.s3     = s3_client
        self.bucket = bucket
        self.key    = key
        self._pos   = 0

        head = s3_client.head_object(Bucket=bucket, Key=key)
        self._size = head["ContentLength"]
        log.info(f"Tamaño ZIP: {self._size/(1024**3):.2f} GB")

    def read(self, n=-1):
        if n == -1:
            end = self._size - 1
        else:
            end = min(self._pos + n - 1, self._size - 1)

        if self._pos > end:
            return b""

        resp = self.s3.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={self._pos}-{end}"
        )
        data = resp["Body"].read()
        self._pos += len(data)
        return data

    def seek(self, pos, whence=0):
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def readable(self):
        return True


# ── Main ─────────────────────────────────────────────────
def explorar():
    log.info("🔍 Explorando ZIP en S3 (filtrado)...")

    s3 = boto3.client("s3", region_name=REGION)
    stream = S3ZipStream(s3, BUCKET, KEY)

    resultados = []

    with zipfile.ZipFile(stream) as z:
        for info in z.infolist():

            ruta = info.filename

            # Solo archivos dentro de testeo_ligas_europa/
            if not ruta.startswith("testeo_ligas_europa/"):
                continue

            # Quitar carpetas
            if ruta.endswith("/"):
                continue

            # 🔥 CLAVE: evitar subcarpetas
            # Contar niveles
            partes = ruta.split("/")

            # Queremos SOLO: testeo_ligas_europa/archivo
            if len(partes) != 2:
                continue

            ext = ruta.split(".")[-1].lower() if "." in ruta else "sin_ext"

            resultados.append({
                "ruta"      : ruta,
                "nombre"    : partes[-1],
                "size_mb"   : info.file_size / (1024**2),
                "zip_mb"    : info.compress_size / (1024**2),
                "ext"       : ext
            })

    # ── Ordenar por tamaño
    resultados.sort(key=lambda x: x["size_mb"], reverse=True)

    log.info(f"📂 Archivos fuera de 'testeo_ligas_europa': {len(resultados)}")

    for r in resultados[:20]:  # top 20
        log.info(f"{r['size_mb']:.1f} MB | {r['ext']} | {r['ruta']}")

    # ── Guardar CSV
    output_csv = "archivos_filtrados.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

    log.info(f"📄 CSV generado: {output_csv}")
    log.info(f"📝 Log: {log_filename}")


if __name__ == "__main__":
    try:
        explorar()
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)