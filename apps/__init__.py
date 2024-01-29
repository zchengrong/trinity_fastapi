import logging.config

from fastapi import FastAPI

from apps.apis.routers import router
from config.logging_env import LOGGER_CONFIG_DICT


def create_app():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    logging.config.dictConfig(LOGGER_CONFIG_DICT)
    return app
