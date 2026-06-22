# extraer_ligas_seleccionadas_s3.py
import boto3
import zipfile
import io
import logging
import sys
from datetime import datetime

# ── Logging ──────────────────────────────────────────────
log_filename = f"extraer_ligas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────
BUCKET     = "rayo-scout-data"
KEY_ZIP    = "europa_temp.zip"
REGION     = "eu-west-1"
CARPETA_S3 = "ligas_seleccionadas"
CHUNK_SIZE = 50 * 1024 * 1024  # 50MB

# ── Temporadas y ligas a extraer ─────────────────────────
TEMPORADAS = ["2024-2025", "2025-2026"]

LIGAS = [
    # España
    "Spain_Primera_Division",       # LaLiga
    "Spain_Segunda_Division",       # LaLiga 2
    # Inglaterra
    #"England_Premier_League",       # Premier League
    #"England_Championship",         # Championship (2ª)
    # Alemania
    #"Germany_Bundesliga",           # Bundesliga
    #"Germany_2_Bundesliga",         # 2. Bundesliga
    # Italia
    #"Italy_Serie_A",                # Serie A
    #"Italy_Serie_B",                # Serie B
    # Francia
    #"France_Ligue_1",               # Ligue 1
    #"France_Ligue_2",               # Ligue 2
    # Portugal
    #"Portugal_Primeira_Liga",       # Primeira Liga
    # ⚠️ No existe Liga_Portugal (2ª) en el ZIP — solo hay Primeira Liga
    # Si aparece en el futuro se llamaría "Portugal_Liga_Portugal_2"
]

# ── Stream por rangos de bytes ────────────────────────────
class S3ZipStream:
    def __init__(self, s3_client, bucket, key):
        self.s3    = s3_client
        self.bucket = bucket
        self.key   = key
        self._pos  = 0
        head       = s3_client.head_object(Bucket=bucket, Key=key)
        self._size = head["ContentLength"]

    def read(self, n=-1):
        end = (self._size - 1) if n == -1 else min(self._pos + n - 1, self._size - 1)
        if self._pos > end:
            return b""
        resp = self.s3.get_object(
            Bucket=self.bucket, Key=self.key,
            Range=f"bytes={self._pos}-{end}"
        )
        data = resp["Body"].read()
        self._pos += len(data)
        return data

    def seek(self, pos, whence=0):
        if whence == 0:   self._pos = pos
        elif whence == 1: self._pos += pos
        elif whence == 2: self._pos = self._size + pos
        return self._pos

    def tell(self):     return self._pos
    def seekable(self): return True
    def readable(self): return True

# ── Subida a S3 (directa o multipart según tamaño) ───────
def subir_a_s3(s3, bucket, key, datos):
    if len(datos) < 5 * 1024 * 1024:
        s3.put_object(Bucket=bucket, Key=key, Body=datos)
    else:
        mpu       = s3.create_multipart_upload(Bucket=bucket, Key=key)
        upload_id = mpu["UploadId"]
        parts     = []
        try:
            for i, offset in enumerate(range(0, len(datos), CHUNK_SIZE), 1):
                chunk = datos[offset:offset + CHUNK_SIZE]
                resp  = s3.upload_part(Bucket=bucket, Key=key,
                                       PartNumber=i, UploadId=upload_id, Body=chunk)
                parts.append({"PartNumber": i, "ETag": resp["ETag"]})
            s3.complete_multipart_upload(
                Bucket=bucket, Key=key,
                MultipartUpload={"Parts": parts}, UploadId=upload_id
            )
        except Exception as e:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            raise e

# ── Función de filtro ─────────────────────────────────────
def debe_incluir(filename):
    """Devuelve True solo si el archivo pertenece a una liga
    y temporada de interés, excluyendo checkpoints y carpetas."""
    if filename.endswith("/"):
        return False
    if ".ipynb_checkpoints" in filename:
        return False
    tiene_liga      = any(liga in filename for liga in LIGAS)
    tiene_temporada = any(t in filename for t in TEMPORADAS)
    return tiene_liga and tiene_temporada

# ── Main ─────────────────────────────────────────────────
def extraer():
    log.info("="*65)
    log.info("EXTRACTOR — 5 GRANDES LIGAS + PORTUGAL · 2024-25 / 2025-26")
    log.info(f"Origen  : s3://{BUCKET}/{KEY_ZIP}")
    log.info(f"Destino : s3://{BUCKET}/{CARPETA_S3}/")
    log.info("Ligas incluidas:")
    for liga in LIGAS:
        log.info(f"  · {liga}")
    log.info(f"Temporadas: {TEMPORADAS}")
    log.info("="*65)

    # 1. Conectar
    log.info("Paso 1: Conectando a S3...")
    s3 = boto3.client("s3", region_name=REGION)
    log.info("Conexión establecida")

    # 2. Abrir ZIP por rangos
    log.info("Paso 2: Abriendo ZIP por rangos (sin descargar)...")
    stream = S3ZipStream(s3, BUCKET, KEY_ZIP)
    log.info(f"ZIP en S3: {stream._size/(1024**3):.2f} GB")

    # 3. Filtrar
    log.info("Paso 3: Filtrando archivos relevantes...")
    with zipfile.ZipFile(stream) as z:
        todos     = z.infolist()
        filtrados = [f for f in todos if debe_incluir(f.filename)]

        # Resumen de qué hay por liga
        log.info(f"Total archivos en ZIP : {len(todos):,}")
        log.info(f"Archivos a extraer    : {len(filtrados):,}")
        total_mb = sum(f.file_size for f in filtrados) / (1024**2)
        log.info(f"Tamaño estimado       : {total_mb:.0f} MB")

        log.info("\nDesglose por liga:")
        for liga in LIGAS:
            n = sum(1 for f in filtrados if liga in f.filename)
            mb = sum(f.file_size for f in filtrados if liga in f.filename) / (1024**2)
            log.info(f"  {liga:<40} {n:>5} archivos — {mb:.0f} MB")
        log.info("="*65)

        # 4. Extraer y subir
        log.info("Paso 4: Extrayendo y subiendo a S3...")
        subidos    = 0
        errores    = 0
        mb_subidos = 0
        total      = len(filtrados)

        for i, info in enumerate(filtrados, 1):
            destino_key = f"{CARPETA_S3}/{info.filename}"
            try:
                datos = z.read(info.filename)
                subir_a_s3(s3, BUCKET, destino_key, datos)
                mb_subidos += len(datos) / (1024**2)
                subidos    += 1
                log.info(f"[{i}/{total}] ✅ {info.filename} ({len(datos)/1024:.1f} KB)")
            except Exception as e:
                errores += 1
                log.error(f"[{i}/{total}] ❌ Error: {info.filename} — {e}")

        # 5. Resumen final
        log.info("="*65)
        log.info("RESUMEN FINAL:")
        log.info(f"  Archivos subidos : {subidos:,}")
        log.info(f"  Errores          : {errores}")
        log.info(f"  MB subidos a S3  : {mb_subidos:.1f} MB")
        log.info(f"  Destino          : s3://{BUCKET}/{CARPETA_S3}/")
        log.info("="*65)
        log.info(f"Log guardado en: {log_filename}")
        log.info("✅ Extracción completada")

if __name__ == "__main__":
    try:
        extraer()
    except Exception as e:
        log.critical(f"Error crítico: {e}")
        log.info(f"Revisa el log: {log_filename}")
        sys.exit(1)