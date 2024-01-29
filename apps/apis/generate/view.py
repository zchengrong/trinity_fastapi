import logging
import time

from apps.apis.generate.model import GenerateImageModel, GenerateCancelModel
from apps.apis.routers import router
from apps.services.generate.generate_triton_server import GenerateImage


@router.post("/generate_image")
async def generate_image(data: GenerateImageModel):
    logging.info(data)
    service = GenerateImage(data)
    return {"result": await service.get_result()}


@router.post("/generate_cancel")
async def generate_image(data: GenerateCancelModel):
    logging.info(data)
    service = GenerateImage(data)
    return {"result": service.get_result()}
