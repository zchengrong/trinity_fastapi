import os

MINIO_IP = os.getenv("MINIO_IP", "www.minio.aida.com.hk")
MINIO_PORT = os.getenv("MINIO_PORT", 9000)
MINIO_ACCESS = os.getenv("MINIO_ACCESS", 'vXKFLSJkYeEq2DrSZvkB')
MINIO_SECRET = os.getenv("MINIO_SECRET", 'uKTZT3x7C43WvPN9QTc99DiRkwddWZrG9Uh3JVlR')
MINIO_SECURE = os.getenv("MINIO_SECURE", True)
