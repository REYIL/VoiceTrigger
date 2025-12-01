import logging
import sys
from colorlog import ColoredFormatter


class ColorLogger:
    """
    Simple colored logger wrapper. Uses English messages only.
    """
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    DEFAULT_COLORS = {
        "DEBUG": "cyan",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white"
    }

    def __init__(self, name="AppLogger", level="info", colors=None):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(self.LEVELS.get(level, logging.INFO))

        if colors is None:
            colors = self.DEFAULT_COLORS

        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)s | %(message)s",
            log_colors=colors
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def set_level(self, level: str):
        self.logger.setLevel(self.LEVELS.get(level, logging.INFO))

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        # convenience wrapper for full stacktrace logging
        self.logger.exception(msg, *args, **kwargs)
