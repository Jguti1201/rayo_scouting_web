# download_and_upload_s3.py
import boto3
import requests
import os
from pathlib import Path

# ── Config ───────────────────────────────────────────────
FILE_URL = "https://dl-bkgz5xt6.swisstransfer.com/api/download/201e2b44-5595-4a71-95b5-a27a6ddc31ec/33a2d743-ae31-46a1-a285-dceefcddbae4"

BUCKET = "rayo-scout-data"
S3_KEY = "europa.zip"

AWS_ACCESS_KEY = "TU_ACCESS_KEY_ID"
AWS_SECRET_KEY = "TU_SECRET_ACCESS_KEY"
AWS_REGION = "eu-west-1"

# Carpeta temporal de descarga
LOCAL_PATH = Path(r"C:\Users\jaime\Downloads\europa_temp.zip")

# ── Descarga a disco ─────────────────────────────────────
print("⬇️  Descargando desde SwissTransfer...")

headers_download = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.swisstransfer.com/"
}

with requests.get(FILE_URL, stream=True, timeout=300, headers=headers_download) as r:
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    descargado = 0

    with open(LOCAL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):  # 10MB chunks
            if chunk:
                f.write(chunk)
                descargado += len(chunk)
                if total:
                    pct = descargado / total * 100
                    print(f"  Descargado: {descargado/(1024**3):.2f} GB / {total/(1024**3):.2f} GB ({pct:.1f}%)", end="\r")

print(f"\n✅ Descarga completa: {LOCAL_PATH}")

# ── Subida a S3 con multipart ────────────────────────────
print("\n⬆️  Subiendo a S3...")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

file_size = LOCAL_PATH.stat().st_size
print(f"Tamaño del archivo: {file_size/(1024**3):.2f} GB")

# Multipart upload
mpu = s3.create_multipart_upload(Bucket=BUCKET, Key=S3_KEY)
upload_id = mpu["UploadId"]

chunk_size = 100 * 1024 * 1024  # 100MB por parte
parts = []
part_num = 1
total_subido = 0

try:
    with open(LOCAL_PATH, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            print(f"  Subiendo parte {part_num} — {total_subido/(1024**3):.2f} GB subidos...", end="\r")

            resp = s3.upload_part(
                Bucket=BUCKET, Key=S3_KEY,
                PartNumber=part_num, UploadId=upload_id,
                Body=chunk
            )
            parts.append({"PartNumber": part_num, "ETag": resp["ETag"]})
            total_subido += len(chunk)
            part_num += 1

    s3.complete_multipart_upload(
        Bucket=BUCKET, Key=S3_KEY,
        MultipartUpload={"Parts": parts},
        UploadId=upload_id
    )
    print(f"\n✅ Subida completa a s3://{BUCKET}/{S3_KEY}")

except Exception as e:
    print(f"\n❌ Error en la subida: {e}")
    s3.abort_multipart_upload(Bucket=BUCKET, Key=S3_KEY, UploadId=upload_id)
    raise

# ── Borrar archivo local ─────────────────────────────────
print("\n🗑️  Borrando archivo temporal del PC...")
LOCAL_PATH.unlink()
print(f"✅ Borrado: {LOCAL_PATH}")
print("\n🎉 Todo listo — europa.zip está en S3 y tu PC está limpio")
