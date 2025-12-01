"""
calibration.py

Класс VoiceCalibrator — модуль для калибровки уровней голоса: тихо, нормально, громко.
Делает повторные записи, вычисляет RMS и HF-спектр, усредняет результаты
и сохраняет их в voice_calibration.json.

Использование:

    from calibration import VoiceCalibrator
    VoiceCalibrator.calibrate()

"""

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import sounddevice as sd


class VoiceCalibrator:
    OUT_FILE = Path(__file__).parent / "voice_calibration.json"

    # -----------------------------------------------------------------------
    #                               AUDIO UTILS
    # -----------------------------------------------------------------------

    @staticmethod
    def record_seconds(seconds: float, samplerate: int = 16000,
                       channels: int = 1) -> np.ndarray:
        """Записывает звук и возвращает float32 [-1..1]."""
        print(f"Recording {seconds:.1f}s... speak now.")
        frames = int(seconds * samplerate)

        rec = sd.rec(frames, samplerate=samplerate,
                     channels=channels, dtype='int16')
        sd.wait()

        if rec.ndim > 1:
            rec = rec.mean(axis=1)

        return rec.astype(np.float32) / 32768.0

    @staticmethod
    def windowed_rms_db(data: np.ndarray, samplerate: int,
                        win_sec: float = 0.2) -> np.ndarray:
        """RMS (dB) по окнам."""
        win_len = max(1, int(win_sec * samplerate))
        n = len(data)

        if n < win_len:
            rms = np.sqrt(np.mean(np.square(data)))
            return np.array([20 * np.log10(rms + 1e-12)])

        result = []
        for start in range(0, n - win_len + 1, win_len):
            w = data[start:start + win_len]
            rms = np.sqrt(np.mean(np.square(w)))
            result.append(20 * np.log10(rms + 1e-12))

        return np.array(result)

    @staticmethod
    def windowed_hf_ratio(data: np.ndarray, samplerate: int,
                          win_sec: float = 0.2) -> np.ndarray:
        """Отношение HF/LF по окнам."""
        from numpy.fft import rfft, rfftfreq

        win_len = max(4, int(win_sec * samplerate))
        n = len(data)

        def hf_ratio(arr):
            fft = rfft(arr)
            mag = np.abs(fft)
            freqs = rfftfreq(len(arr), 1 / samplerate)
            low = mag[freqs < 1000].sum() + 1e-12
            high = mag[freqs >= 1000].sum() + 1e-12
            return high / low

        if n < win_len:
            return np.array([hf_ratio(data)])

        result = []
        for start in range(0, n - win_len + 1, win_len):
            w = data[start:start + win_len]
            result.append(hf_ratio(w))

        return np.array(result)

    @staticmethod
    def summarize_sample(data: np.ndarray, sr: int) -> Dict[str, float]:
        """Вычисляет статистики для одного аудиофрагмента."""
        dbs = VoiceCalibrator.windowed_rms_db(data, sr)
        hfs = VoiceCalibrator.windowed_hf_ratio(data, sr)

        return {
            "db_mean": float(np.mean(dbs)),
            "db_std": float(np.std(dbs)),
            "hf_mean": float(np.mean(hfs)),
            "hf_std": float(np.std(hfs)),
            "windows": int(len(dbs))
        }

    # -----------------------------------------------------------------------
    #                           HIGH LEVEL LOGIC
    # -----------------------------------------------------------------------

    @staticmethod
    def _multi_record(level_name: str, repeats: int, seconds: float,
                       samplerate: int, interactive: bool = True) -> Dict[str, float]:
        """Делает несколько записей уровня громкости и усредняет результаты."""
        samples = []

        for i in range(1, repeats + 1):
            print(f"\nЗапись {i}/{repeats} для уровня '{level_name}'")

            if interactive:
                input(f"Нажми Enter и говори {level_name} ...")
                time.sleep(0.3)

            data = VoiceCalibrator.record_seconds(seconds, samplerate)
            samples.append(VoiceCalibrator.summarize_sample(data, samplerate))

        # усреднение
        avg = {
            "db_mean": float(np.mean([s["db_mean"] for s in samples])),
            "db_std": float(np.mean([s["db_std"] for s in samples])),
            "hf_mean": float(np.mean([s["hf_mean"] for s in samples])),
            "hf_std": float(np.mean([s["hf_std"] for s in samples])),
            "windows": int(sum(s["windows"] for s in samples))
        }

        return avg

    # -----------------------------------------------------------------------
    #                             PUBLIC API
    # -----------------------------------------------------------------------

    @staticmethod
    def calibrate(
        samplerate: int = 16000,
        repeats: int = 3,
        seconds: float = 4.0,
        interactive: bool = True,
        calibration_path: Path = None
    ) -> Dict:
        """
        Основная функция калибровки.
        """
        outfile = calibration_path or VoiceCalibrator.OUT_FILE

        print("=== Калибровка голоса ===")

        quiet = VoiceCalibrator._multi_record("Шёпот", repeats, seconds, samplerate, interactive)
        normal = VoiceCalibrator._multi_record("Нормальная речь", repeats, seconds, samplerate, interactive)
        loud = VoiceCalibrator._multi_record("Крик", repeats, seconds, samplerate, interactive)

        out = {
            "samplerate": samplerate,
            "seconds_per_sample": seconds,
            "repeats": repeats,
            "quiet": quiet,
            "normal": normal,
            "loud": loud,
            "created_at": time.time()
        }

        outfile.write_text(json.dumps(out, indent=2, ensure_ascii=False))

        print(f"\nКалибровка сохранена в {outfile.resolve()}")

        print("Средние dB:")
        print(f" quiet:  {quiet['db_mean']:.2f}")
        print(f" normal: {normal['db_mean']:.2f}")
        print(f" loud:   {loud['db_mean']:.2f}")

        print(
            "HF:",
            f"{quiet['hf_mean']:.2f}",
            f"{normal['hf_mean']:.2f}",
            f"{loud['hf_mean']:.2f}"
        )

        return out


# Позволяет запускать напрямую:
if __name__ == "__main__":
    VoiceCalibrator.calibrate()
