import asyncio
import time
import threading
from typing import Callable, Optional, List, Union, Any, Coroutine
import inspect
from pathlib import Path

import numpy as np
import sounddevice as sd

from ..utils.logger import ColorLogger
from ..utils.filter import Filter, TextContext
from ..utils.levenshtein import is_match_by_lev
from .vldetector import VoiceLevelDetector
from .asmanager import AudioStreamManager
from .speechr import SpeechRecognizer

try:
    import noisereduce as nr
    _NOISEREDUCE_AVAILABLE = True
except Exception:
    _NOISEREDUCE_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


class VoiceTrigger:
    def __init__(self, model_path,
                 sample_rate=16000, blocksize=2000,
                 keywords: Optional[List[str]] = None,
                 quick_words: Optional[List[str]] = None,
                 buffer_time_seconds=6, buffer_max_chars=1000,
                 noise_reduction=False, hp_cutoff=100,
                 batch_blocks=6,
                 voice_detector_buffer_seconds=5,
                 calibration_path: Optional[Path] = None,
                 rms_thresholds=None,
                 device: Optional[Union[int, str]] = None,
                 logger: Optional[ColorLogger] = None):
        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self.log = logger or ColorLogger(level="info")

        # speech recognizer wrapper
        self.speech = SpeechRecognizer(model_path=model_path, sample_rate=sample_rate, logger=self.log)

        # initial lists (unique)
        self.keywords = list(dict.fromkeys(keywords or []))
        self.quick_words = list(dict.fromkeys(quick_words or []))

        # flags
        self.active_main = False
        self.active_kw = False
        self.active_quick = False

        # buffer for dynamic text
        self.text_buffer = []
        self.buffer_active = False
        self.buffer_time_seconds = buffer_time_seconds
        self.buffer_max_chars = buffer_max_chars

        # voice detector
        self.voice_detector = VoiceLevelDetector(
            samplerate=sample_rate,
            blocksize=blocksize,
            buffer_seconds=voice_detector_buffer_seconds,
            calibration_path=calibration_path,
            logger=self.log,
            rms_thresholds=rms_thresholds
        )

        # audio manager (store device selection)
        self.device = device
        self.audio_manager = AudioStreamManager(
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype='int16',
            channels=1,
            callback=self._audio_callback,
            device=self.device,
            logger=self.log
        )

        # async queue/loop/stream
        self.async_q = None
        self.loop = None
        self._lock = threading.Lock()

        self._silence_start_main = time.time()
        self._silence_start_kw = time.time()
        self.latest_text = ""

        if (not _NOISEREDUCE_AVAILABLE or not _SCIPY_AVAILABLE) and noise_reduction:
            raise Exception("To enable `noise_reduction=True`, `noisereduce` and `scipy` are missing. Please install them with `pip install noisereduce scipy`.")
        elif noise_reduction or not noise_reduction:
            self.noise_reduction = noise_reduction

        self.hp_cutoff = hp_cutoff
        self.noise_floor_db = -50.0

        # batch
        self.batch_blocks = max(1, int(batch_blocks))

        # callbacks
        self._text_handlers = []
        self._keyword_handlers = []
        self._quick_handlers = []
        self._silence_handlers = []
        self._kw_silence_handlers = []

        self._handlers_lock = threading.Lock()

        # internal state for last matched quick/keyword
        self._last_keyword = None
        self._last_quick = None

        # For test mode: external injection of raw audio bytes (bypass stream)
        self._test_mode = False

    # ----------- Public decorator API (keeps original API) -----------
    def text(self, flt: Optional[Filter] = None):
        if flt is None:
            flt = Filter(None)

        def decorator(func: Callable[[TextContext], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                self._text_handlers.append((flt, func))
            return func
        return decorator

    def keyword(self, flt: Optional[Filter] = None):
        if flt is None:
            flt = Filter(None)

        def decorator(func: Callable[[TextContext], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                if flt.phrases:
                    for p in flt.phrases:
                        if p not in self.keywords:
                            self.keywords.append(p)
                self._keyword_handlers.append((flt, func))
            return func
        return decorator

    def quick(self, flt: Optional[Filter] = None):
        if flt is None:
            flt = Filter(None)

        def decorator(func: Callable[[TextContext], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                if flt.phrases:
                    for p in flt.phrases:
                        if p not in self.quick_words:
                            self.quick_words.append(p)
                self._quick_handlers.append((flt, func))
            return func
        return decorator

    def on_silence(self):
        def decorator(func: Callable[[float], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                self._silence_handlers.append(func)
            return func
        return decorator

    def on_kw_silence(self):
        def decorator(func: Callable[[float], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                self._kw_silence_handlers.append(func)
            return func
        return decorator

    # --------- control operations (start/stop/reload) ----------
    def start_recognition_main(self):
        try:
            self.speech.reset()
            self.active_main = True
            self.active_quick = True
            self.buffer_active = False
            self.log.debug("Main recognition enabled.")
        except Exception:
            self.log.exception("Failed to enable main recognition.")

    def stop_recognition_main(self):
        self.active_main = False
        self.active_quick = False
        self.log.debug("Main recognition disabled.")

    def start_recognition_keywords(self):
        try:
            self.speech.reset()
            self.active_kw = True
            self.log.debug("Keyword recognition enabled.")
        except Exception:
            self.log.exception("Failed to enable keyword recognition.")

    def stop_recognition_keywords(self):
        self.active_kw = False
        self.log.debug("Keyword recognition disabled.")

    def reload_model(self, model_path: Optional[str] = None):
        try:
            self.speech.reload_model(new_model_path=model_path)
        except Exception:
            self.log.exception("Failed to reload model.")

    # --------- device management ----------
    def set_input_device(self, device: Optional[Union[int, str]], restart_stream: bool = False):
        """
        Set input device (index or name). If restart_stream=True and stream is active,
        it will stop and restart stream with the new device.
        """
        self.log.info(f"Set input device to {device!r}")
        self.device = device
        self.audio_manager.device = device
        if restart_stream:
            was_active = self.audio_manager.is_active()
            try:
                if was_active:
                    self.log.info("Restarting audio stream to apply device change...")
                    self.audio_manager.stop()
                    # small delay to ensure OS releases device (non-blocking)
                    time.sleep(0.1)
                    self.audio_manager.start()
                    self.log.info("Audio stream restarted with new device.")
            except Exception:
                self.log.exception("Failed to restart audio stream after device change.")

    @staticmethod
    def list_input_devices() -> List[dict]:
        """
        Returns a list of available input devices (each item: {'index': int, 'name': str, 'max_input_channels': int})
        """
        out = []
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0:
                    out.append({
                        'index': i,
                        'name': d.get('name'),
                        'max_input_channels': d.get('max_input_channels', 0)
                    })
        except Exception:
            # If query fails, return empty but log
            ColorLogger().exception("Failed to query sounddevice devices.")
        return out

    # --------- audio callback (from AudioStreamManager) ----------
    def _audio_callback(self, data_bytes, frames, time_info, status):
        try:
            # Feed voice detector and async queue
            try:
                self.voice_detector.process_block(data_bytes)
            except Exception:
                self.log.debug("Voice detector failed for a block.", exc_info=True)

            if self.loop and self.async_q:
                try:
                    # put bytes to async queue from audio thread
                    self.loop.call_soon_threadsafe(self.async_q.put_nowait, data_bytes)
                except Exception:
                    self.log.debug("Failed to put audio block into async queue.", exc_info=True)
        except Exception:
            self.log.exception("Unhandled error inside audio callback.")

    # --------- preprocessing ----------
    def _highpass_filter(self, data_float: np.ndarray):
        # if _SCIPY_AVAILABLE:
        nyq = 0.5 * self.sample_rate
        normal_cutoff = self.hp_cutoff / nyq
        try:
            b, a = butter(1, normal_cutoff, btype='high', analog=False)
            return filtfilt(b, a, data_float)
        except Exception:
            self.log.debug("SciPy highpass failed; using fallback filter.", exc_info=True)
        # alpha = 0.995
        # y = np.zeros_like(data_float)
        # prev_x = 0.0
        # prev_y = 0.0
        # for i, x in enumerate(data_float):
        #     y[i] = alpha * (prev_y + x - prev_x)
        #     prev_x = x
        #     prev_y = y[i]
        # return y

    def _simple_noise_gate_and_normalize(self, data_float: np.ndarray):
        if data_float.size == 0:
            return data_float
        rms = np.sqrt(np.mean(data_float ** 2))
        db = 20 * np.log10(rms + 1e-12)
        try:
            if db < (self.noise_floor_db):
                self.noise_floor_db = 0.95 * self.noise_floor_db + 0.05 * db
            else:
                self.noise_floor_db = 0.999 * self.noise_floor_db + 0.001 * db
        except Exception:
            self.log.debug("Noise floor update failed.", exc_info=True)

        if db < (self.noise_floor_db + 6.0):
            data_float = data_float * (rms / (rms + 1e-6)) * 0.1
        peak = np.max(np.abs(data_float)) + 1e-12
        if peak > 0.95:
            data_float = data_float / peak * 0.95
        return data_float

    def _preprocess_audio(self, data_bytes: bytes) -> bytes:
        try:
            data_int16 = np.frombuffer(data_bytes, dtype=np.int16)
            data_float = data_int16.astype(np.float32) / 32768.0
            data_float = self._highpass_filter(data_float)
            if self.noise_reduction:
                try:
                    nr_out = nr.reduce_noise(y=data_float, sr=self.sample_rate, stationary=False)
                    data_float = nr_out.astype(np.float32)
                except Exception:
                    self.log.debug("noisereduce failed, falling back to simple gate.", exc_info=True)
                    data_float = self._simple_noise_gate_and_normalize(data_float)
            else:
                data_float = self._simple_noise_gate_and_normalize(data_float)
            clipped = np.clip(data_float, -1.0, 1.0)
            out_int16 = (clipped * 32767.0).astype(np.int16)
            return out_int16.tobytes()
        except Exception:
            self.log.debug("Preprocessing failed for audio block; returning raw bytes.", exc_info=True)
            return data_bytes

    # --------- buffering text ----------
    def _append_to_buffer(self, text: str):
        ts = time.time()
        with self._lock:  # FIX: protect buffer with lock
            self.text_buffer.append((ts, text))
            cutoff = ts - self.buffer_time_seconds
            self.text_buffer = [(t, txt) for (t, txt) in self.text_buffer if t >= cutoff]
            total_chars = sum(len(txt) for (_, txt) in self.text_buffer)
            while total_chars > self.buffer_max_chars and len(self.text_buffer) > 1:
                self.text_buffer.pop(0)
                total_chars = sum(len(txt) for (_, txt) in self.text_buffer)

    def get_buffered_phrase(self):
        if not self.text_buffer:
            return None
        text_all = " ".join(txt for (_, txt) in self.text_buffer)
        for kw in self.keywords:
            idx = text_all.find(kw)
            if idx != -1:
                return text_all[idx:]
        return text_all

    # --------- General matching + dispatch utilities ----------
    def _match_filter_against_text(self, flt: Filter, text: str, voice_mode: str) -> Optional[str]:
        text_low = text.lower()
        if flt.is_wildcard():
            return None
        for pattern in flt.phrases:
            pat_low = pattern.lower()
            if pat_low in text_low:
                return pattern
            words = [w for w in ''.join(ch if ch.isalnum() else ' ' for ch in text_low).split() if w]
            for w in words:
                if is_match_by_lev(w, pat_low, flt.lv):
                    return pattern
            if is_match_by_lev(text_low, pat_low, flt.lv):
                return pattern
        return None

    async def _invoke_handler(self, handler: Callable[..., Any], arg):
        """
        Invoke a handler safely:
        - If it's an async function, schedule it on the event loop.
        - If it's synchronous, run in default executor to avoid blocking the loop.
        """
        try:
            if inspect.iscoroutinefunction(handler):
                # schedule coroutine
                _ = asyncio.create_task(handler(arg))
            else:
                # run sync handler in executor to avoid blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, handler, arg)
        except Exception:
            self.log.exception("Handler invocation failed.")

    async def _dispatch_handlers_generic(self, handlers, text_or_kw: str, voice_mode: str, is_phrase_source=False):
        """
        Generic dispatcher for text/keyword/quick handlers.
        If flt.is_wildcard() -> call with match=None or match=the kw (for keywords/quick).
        """
        ts = time.time()
        with self._handlers_lock:
            handlers_copy = list(handlers)
        for flt, handler in handlers_copy:
            try:
                if flt.mode is not None and voice_mode != flt.mode.value:
                    continue

                if flt.is_wildcard():
                    # wildcard means we accept everything
                    match = text_or_kw if is_phrase_source else None
                    ctx = TextContext(text=text_or_kw, mode=voice_mode, match=match, ts=ts)
                    await self._invoke_handler(handler, ctx)
                    continue

                match = self._match_filter_against_text(flt, text_or_kw, voice_mode)
                if match is not None:
                    ctx = TextContext(text=text_or_kw, mode=voice_mode, match=match, ts=ts)
                    await self._invoke_handler(handler, ctx)
            except Exception:
                self.log.exception("Error while dispatching to handler.")

    # --------- silence invocation helpers ----------
    async def _call_silence_handlers(self, silence_main: float):
        with self._handlers_lock:
            handlers = list(self._silence_handlers)
        for h in handlers:
            try:
                if inspect.iscoroutinefunction(h):
                    _ = asyncio.create_task(h(silence_main))
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, h, silence_main)
            except Exception:
                self.log.exception("Failure while calling silence handler.")

    async def _call_kw_silence_handlers(self, silence_kw: float):
        with self._handlers_lock:
            handlers = list(self._kw_silence_handlers)
        for h in handlers:
            try:
                if inspect.iscoroutinefunction(h):
                    _ = asyncio.create_task(h(silence_kw))
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, h, silence_kw)
            except Exception:
                self.log.exception("Failure while calling keyword silence handler.")

    # --------- main processing loop ----------
    async def process_audio(self):
        """
        Pull from async queue, process audio in batches, run recognizers and dispatch handlers.
        Returns tuple similar to previous implementation.
        """
        main_text = None
        keyword_found = None
        quick_found = None

        if self.async_q is None:
            return None, None, None, time.time() - self._silence_start_main, time.time() - self._silence_start_kw

        while not self.async_q.empty():
            available = self.async_q.qsize()
            n = min(available, self.batch_blocks)
            if n <= 0:
                break
            frames = []
            for _ in range(n):
                try:
                    frames.append(self.async_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if not frames:
                break
            batch_bytes = b"".join(frames)
            processed = self._preprocess_audio(batch_bytes)

            # main recognition
            if self.active_main:
                try:
                    text, partial = self.speech.process_main(processed)
                    # Prefer final text over partials for buffer
                    if text and not (self._last_quick and is_match_by_lev(text, self._last_quick, 10)):
                        main_text = text
                        self.latest_text = text
                        self._append_to_buffer(text)
                        self._silence_start_main = time.time()
                        if self.buffer_active:
                            self.buffer_active = False
                    else:
                        # partial handling: keep buffer and latest_text updated
                        if partial:
                            self.latest_text = partial
                            self._append_to_buffer(partial)
                            self._silence_start_main = time.time()
                except Exception:
                    self.log.exception("Error while running main recognition.")

            # keywords
            if self.active_kw:
                try:
                    _, partial_kw = self.speech.process_kw(processed)
                    if partial_kw:
                        self._silence_start_kw = time.time()
                        for kw in self.keywords:
                            if is_match_by_lev(partial_kw, kw, 10) or kw.lower() in partial_kw.lower():
                                if kw != getattr(self, "_last_keyword", None):
                                    keyword_found = kw
                                    self._last_keyword = kw
                                    self.buffer_active = True
                                    self.text_buffer = [(time.time(), kw)]
                                break
                    else:
                        # reset last keyword if no partial matches
                        if not any(is_match_by_lev(partial_kw, kw, 10) or kw.lower() in partial_kw.lower() for kw in self.keywords):
                            self._last_keyword = None
                except Exception:
                    self.log.exception("Error while running keyword recognition.")

            # quick words
            if self.active_quick:
                try:
                    _, partial_q = self.speech.process_quick(processed)
                    if partial_q:
                        for qw in self.quick_words:
                            if is_match_by_lev(partial_q, qw, 10) or qw.lower() in partial_q.lower():
                                if qw != getattr(self, "_last_quick", None):
                                    quick_found = qw
                                    self._last_quick = qw
                                break
                    else:
                        if not any(is_match_by_lev(partial_q, qw, 10) or qw.lower() in partial_q.lower() for qw in self.quick_words):
                            self._last_quick = None
                except Exception:
                    self.log.exception("Error while running quick-word recognition.")

        silence_time_main = time.time() - self._silence_start_main
        silence_time_kw = time.time() - self._silence_start_kw

        voice_mode = self.voice_detector.get_dominant_level()

        # dispatching via unified generic dispatcher
        if main_text:
            await self._dispatch_handlers_generic(self._text_handlers, main_text, voice_mode, is_phrase_source=False)
        if keyword_found:
            await self._dispatch_handlers_generic(self._keyword_handlers, keyword_found, voice_mode, is_phrase_source=True)
        if quick_found:
            await self._dispatch_handlers_generic(self._quick_handlers, quick_found, voice_mode, is_phrase_source=True)

        # call silence handlers (non-blocking)
        await self._call_silence_handlers(silence_time_main)
        await self._call_kw_silence_handlers(silence_time_kw)

        return main_text, keyword_found, quick_found, silence_time_main, silence_time_kw

    # --------- run loop & streaming ----------
    async def start_stream(self):
        if self._test_mode:
            self.log.info("Test mode active: not starting audio stream.")
            return
        if self.async_q is None:
            self.loop = asyncio.get_running_loop()
            self.async_q = asyncio.Queue()
        try:
            self.audio_manager.start()
            self.log.info("Audio stream started (async).")
        except Exception:
            self.log.exception("Failed to start audio stream.")

    async def stop_stream(self):
        try:
            self.audio_manager.stop()
            if self.async_q:
                # clear queue
                while not self.async_q.empty():
                    try:
                        self.async_q.get_nowait()
                    except Exception:
                        break
            self.log.info("Audio stream stopped (async).")
        except Exception:
            self.log.exception("Error while stopping audio stream.")

    # Allow injecting raw audio blocks for testing (bypasses the actual microphone)
    async def inject_audio_block(self, data_bytes: bytes):
        if self.async_q is None:
            self.async_q = asyncio.Queue()
            self.loop = asyncio.get_running_loop()
        await self.async_q.put(data_bytes)

    async def run(self, initial_keywords_mode=True):
        await self.start_stream()
        if initial_keywords_mode:
            self.start_recognition_keywords()
        else:
            self.start_recognition_main()

        self.log.info("Say something... (Ctrl+C to exit)")

        try:
            while True:
                await asyncio.sleep(0.05)
                await self.process_audio()
        except asyncio.CancelledError:
            self.log.info("Run loop cancelled.")
        except KeyboardInterrupt:
            self.log.info("KeyboardInterrupt received; stopping.")
        finally:
            self.stop_recognition_main()
            self.stop_recognition_keywords()
            await self.stop_stream()
            self.log.info("Stopping speech recognition")
