LOGGER_CONFIG_DICT = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": "logs/info.log",
            "maxBytes": 10485760,
            "backupCount": 50,
            "encoding": "utf8",
        },
        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "simple",
            "filename": "logs/errors.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8",
        },
        "debug_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "logs/debug.log",
            "maxBytes": 10485760,
            "backupCount": 50,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "my_module": {"level": "INFO", "handlers": ["console"], "propagate": "no"}
    },
    "root": {
        "level": "INFO",
        "handlers": ["error_file_handler", "info_file_handler", "debug_file_handler", "console"],
    },
}
