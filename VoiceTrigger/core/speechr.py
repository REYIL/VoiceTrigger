from typing import Optional
import vosk
import json

from VoiceTrigger.utils.logger import ColorLogger


class SpeechRecognizer:
    """
    Wraps Vosk Model and multiple KaldiRecognizer instances for main/keyword/quick.
    Provides safe methods to accept waveform and get results.
    """
    def __init__(self, model_path: str, sample_rate: int = 16000, logger: Optional[ColorLogger] = None):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model = None
        self.logger = logger or ColorLogger()
        self._load_model()

        # recognizers (can be re-created if model changes)
        self.rec_main = self._make_recognizer()
        self.rec_kw = self._make_recognizer()
        self.rec_quick = self._make_recognizer()

    def _load_model(self):
        try:
            self.model = vosk.Model(self.model_path)
            self.logger.info(f"Vosk model loaded from {self.model_path}")
        except Exception as e:
            self.model = None
            self.logger.exception(f"Failed to load Vosk model from {self.model_path}: {e}")
            raise RuntimeError(f"Cannot load Vosk model from {self.model_path}")  # FIX: raise instead of exit

    def _make_recognizer(self):
        if self.model:
            try:
                return vosk.KaldiRecognizer(self.model, self.sample_rate)
            except Exception as e:
                self.logger.exception(f"Failed to create KaldiRecognizer: {e}")
                return None
        return None

    def reload_model(self, new_model_path: Optional[str] = None):
        if new_model_path:
            self.model_path = new_model_path
        self._load_model()
        self.rec_main = self._make_recognizer()
        self.rec_kw = self._make_recognizer()
        self.rec_quick = self._make_recognizer()
        self.logger.info("Model reloaded.")

    def reset(self):
        try:
            if self.rec_main:
                self.rec_main.Reset()
            if self.rec_kw:
                self.rec_kw.Reset()
            if self.rec_quick:
                self.rec_quick.Reset()
        except Exception as e:
            self.logger.exception(f"Error while resetting recognizers: {e}")

    # Helper accept/process methods: returns tuple(result_text, partial_text)
    def process_main(self, processed_bytes: bytes):
        text = ""
        partial = ""
        try:
            if self.rec_main and self.rec_main.AcceptWaveform(processed_bytes):
                result = json.loads(self.rec_main.Result())
                text = result.get("text", "")
            else:
                partial = json.loads(self.rec_main.PartialResult()).get("partial", "")
        except Exception as e:
            self.logger.debug(f"Error in main recognizer processing: {e}", exc_info=True)
        return text, partial

    def process_kw(self, processed_bytes: bytes):
        # kw recognizer: mainly partials for keyword detection
        try:
            if self.rec_kw and self.rec_kw.AcceptWaveform(processed_bytes):
                _ = json.loads(self.rec_kw.Result())
            partial = json.loads(self.rec_kw.PartialResult()).get("partial", "")
            return "", partial
        except Exception as e:
            self.logger.debug(f"Error in keyword recognizer processing: {e}", exc_info=True)
            return "", ""

    def process_quick(self, processed_bytes: bytes):
        try:
            if self.rec_quick and self.rec_quick.AcceptWaveform(processed_bytes):
                _ = json.loads(self.rec_quick.Result())
            partial = json.loads(self.rec_quick.PartialResult()).get("partial", "")
            return "", partial
        except Exception as e:
            self.logger.debug(f"Error in quick recognizer processing: {e}", exc_info=True)
            return "", ""
