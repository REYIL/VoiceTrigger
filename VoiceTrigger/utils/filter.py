from enum import Enum
from typing import Optional, List, Union


class Mode(Enum):
    whisper = "whisper"
    normal = "normal"
    shout = "shout"


class Filter:
    def __init__(self, phrases: Union[str, List[str], None] = None,
                 lv: int = 10, mode: Optional[Mode] = None):
        if isinstance(phrases, str):
            self.phrases = [phrases]
        elif isinstance(phrases, list):
            self.phrases = list(phrases)
        else:
            self.phrases = []
        self.lv = max(0, int(lv))
        self.mode = mode

    def is_wildcard(self):
        return len(self.phrases) == 0


class TextContext:
    def __init__(self, text: str, mode: str, match: Optional[str], ts: float):
        self.text = text
        self.mode = mode
        self.match = match
        self.timestamp = ts