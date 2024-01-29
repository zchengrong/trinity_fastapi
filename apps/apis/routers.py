from fastapi import APIRouter

router = APIRouter()
from .generate import view
from .design import view
