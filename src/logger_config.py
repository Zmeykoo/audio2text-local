import logging
import warnings


def configure_logging(debug_mode: bool = False, SUPER_debug_mode: bool = False):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler for logs.log
    if debug_mode or SUPER_debug_mode:
        file_handler = logging.FileHandler("logs.log", mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    noisy_loggers = [  # TODO: add FutureWarnings and so on
        "urllib3",
        "numba",
        "httpx",
        "httpx2",
        "huggingface_hub",
    ]
    if not SUPER_debug_mode:
        root_logger.info("disable low level loggers")
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.addHandler(console_handler)
