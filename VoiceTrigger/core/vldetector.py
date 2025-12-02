from collections import deque
from pathlib import Path
from typing import Optional
import json
import threading
import numpy as np

from VoiceTrigger.utils.logger import ColorLogger


class VoiceLevelDetector:
    """
    Adaptive Voice Level Detector.

    - Accepts `calibration_path` for flexibility.
    - Returns dominant level from recent buffer.
    """
    DEFAULT_CALIB_FILE = Path("voice_calibration.json")

    def __init__(self,
                 samplerate=16000,
                 blocksize=2000,
                 rms_thresholds=None,
                 hf_ratio_threshold=1.5,
                 silence_db=-45,
                 buffer_seconds=None,
                 compute_every_n_blocks=1,
                 hf_weight=0.12,
                 calibration_path: Optional[Path] = None,
                 logger: Optional[ColorLogger] = None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.hf_ratio_threshold = hf_ratio_threshold
        self.silence_db = silence_db

        self.rms_thresholds = rms_thresholds or {
            "whisper": -43.0,
            "normal": -15.0,
            "shout": 0.0
        }

        self.hf_weight = float(hf_weight)

        if buffer_seconds is not None:
            approx_blocks = max(1, int((samplerate * buffer_seconds) / max(1, blocksize)))
            maxlen = approx_blocks
        else:
            maxlen = 50

        self.audio_buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()

        self.compute_every_n_blocks = max(1, int(compute_every_n_blocks))
        self._blocks_since_compute = self.compute_every_n_blocks
        self._last_computed_level = "normal"

        self._calib = None
        self.calibration_path = calibration_path or self.DEFAULT_CALIB_FILE
        self.logger = logger or ColorLogger()
        self._load_calibration_if_exists(self.calibration_path)

    def _load_calibration_if_exists(self, path: Path):
        if path and path.exists():
            try:
                raw = json.loads(path.read_text(encoding='utf-8'))
                self._calib = raw
                self._compute_thresholds_from_calib(raw)
                qhf = raw.get("quiet", {}).get("hf_mean")
                nhf = raw.get("normal", {}).get("hf_mean")
                if qhf is not None and nhf is not None:
                    self.hf_ratio_threshold = max(1.0, float(qhf + 0.6 * (nhf - qhf)))
                self.logger.info(f"Voice calibration loaded from {path}")
            except Exception as e:
                self.logger.exception(f"Failed to load calibration from {path}: {e}")
                self._calib = None

    def _compute_thresholds_from_calib(self, calib: dict):
        try:
            q = float(calib["quiet"]["db_mean"])
            n = float(calib["normal"]["db_mean"])
            l = float(calib["loud"]["db_mean"])
        except Exception as e:
            self.logger.debug(f"Incomplete calibration data; skipping threshold computation: {e}")
            return

        if not (q <= n <= l):
            sorted_vals = sorted([q, n, l])
            q, n, l = sorted_vals

        whisper_thr = q + max(1.5, (n - q) * 0.25)
        normal_thr = n + max(2.0, (l - n) * 0.5)

        if whisper_thr >= normal_thr - 1.0:
            whisper_thr = n - max(1.0, (n - q) * 0.2)

        self.rms_thresholds = {
            "whisper": float(whisper_thr),
            "normal": float(normal_thr),
            "shout": float(normal_thr)
        }

        self.silence_db = min(self.silence_db, q - 6.0)

    @staticmethod
    def rms_db(data: np.ndarray):
        rms = np.sqrt(np.mean(np.square(data)))
        db = 20 * np.log10(rms + 1e-6)
        return rms, db

    def hf_ratio(self, data: np.ndarray):
        try:
            fft = np.fft.rfft(data)
            mag = np.abs(fft)
            freqs = np.fft.rfftfreq(len(data), 1 / self.samplerate)
            low = mag[freqs < 1000].sum() + 1e-6
            high = mag[freqs >= 1000].sum() + 1e-6
            return float(high / low)
        except Exception as e:
            self.logger.error(e)
            return 1.0

    def _decide_by_hybrid(self, db: float, hf: float) -> str:
        whisper_thr = self.rms_thresholds["whisper"]
        normal_thr = self.rms_thresholds["normal"]

        if db < self.silence_db:
            return "silence"
        if db < whisper_thr:
            return "whisper"
        if db >= normal_thr:
            return "shout"

        denom = (normal_thr - whisper_thr) if (normal_thr - whisper_thr) != 0 else 1.0
        db_norm = float(np.clip((db - whisper_thr) / denom, 0.0, 1.0))

        hf_base = 1.0
        hf_thr = max(self.hf_ratio_threshold, hf_base + 0.01)
        hf_score = float(np.clip((hf - hf_base) / (hf_thr - hf_base), 0.0, 1.0))

        alpha = 1.0 - self.hf_weight
        combined = alpha * db_norm + (1.0 - alpha) * hf_score

        if combined < 0.25:
            return "whisper"
        elif combined < 0.75:
            return "normal"
        else:
            return "shout"

    def process_block(self, data_bytes: bytes):
        try:
            data = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            # attempt to handle if bytes are float32 array or numpy array passed
            try:
                arr = np.frombuffer(data_bytes, dtype=np.float32)
                data = arr
            except Exception as e:
                self.logger.debug(f"Failed to interpret audio bytes for voice level calculation: {e}", exc_info=True)
                return

        self._blocks_since_compute += 1
        if self._blocks_since_compute >= self.compute_every_n_blocks:
            self._blocks_since_compute = 0
            try:
                rms, db = self.rms_db(data)
                hf = self.hf_ratio(data)
                level = self._decide_by_hybrid(db, hf)
                self._last_computed_level = level
            except Exception as e:
                self.logger.exception(f"Error while computing voice level: {e}")
                level = self._last_computed_level
        else:
            level = self._last_computed_level

        with self._lock:
            self.audio_buffer.append(level)

    def get_dominant_level(self):
        with self._lock:
            counts = {}
            for lvl in self.audio_buffer:
                if lvl != "silence":
                    counts[lvl] = counts.get(lvl, 0) + 1
        if not counts:
            return "normal"
        return max(counts, key=counts.get)
