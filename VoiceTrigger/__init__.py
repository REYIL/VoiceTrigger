from importlib.metadata import metadata

from .utils.logger import ColorLogger
from .utils.filter import Filter, Mode
from .utils.filter import TextContext

from .core.decorators import VoiceTrigger

from .services.calibration import VoiceCalibrator

__all__ = ["ColorLogger", "Filter", "Mode", "TextContext", "VoiceTrigger", "VoiceCalibrator"]

try:
    meta = metadata("VoiceTrigger")
    __version__ = meta.get("Version")
    __author__ = meta.get("Author")
    __license__ = meta.get("License")
    __description__ = meta.get("Summary")
except Exception:
    __version__ = "x.x.x"
    __author__ = "REYIL"
    __license__ = "MIT"
    __description__ = "undefined"
