# src/logging_config.py
import logging
import sys
from logging import Logger


def setup_logging(level: str = None) -> None:
    """
    Настраивает корневой логгер: вывод в stdout, простой формат, уровень.
    """
    lvl = level or logging.INFO
    if isinstance(lvl, str):
        lvl = logging.getLevelName(lvl.upper())

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> Logger:
    """
    Удобная функ- я для получения логгера модуля.
    """
    return logging.getLogger(name)
