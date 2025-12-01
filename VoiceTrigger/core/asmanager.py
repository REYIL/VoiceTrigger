import threading
from typing import Optional, Union
import sounddevice as sd

from VoiceTrigger.utils.logger import ColorLogger


class AudioStreamManager:
    """
    Wraps sounddevice RawInputStream and handles starting/stopping and restart attempts.
    Allows specifying device (index or name).
    """
    def __init__(self, samplerate=16000, blocksize=2000, dtype='int16', channels=1, callback=None, device: Optional[Union[int, str]] = None, logger: Optional[ColorLogger] = None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype
        self.channels = channels
        self.callback = callback
        self._stream = None
        self._lock = threading.Lock()
        self.logger = logger or ColorLogger()
        self.device = device  # device index or name, passed to sounddevice

    def _wrap_callback(self, indata, frames, time_info, status):
        if status:
            self.logger.debug(f"Audio stream status: {status}")
        try:
            # If raw stream returns bytes, pass as-is; if numpy array, convert to bytes
            data_bytes = indata if isinstance(indata, (bytes, bytearray)) else bytes(indata)
        except Exception:
            # fallback: try to get buffer
            try:
                data_bytes = indata.tobytes()
            except Exception:
                self.logger.debug("Failed to convert audio input to bytes.", exc_info=True)
                return
        if self.callback:
            try:
                self.callback(data_bytes, frames, time_info, status)
            except Exception:
                self.logger.exception("Error in user audio callback.")

    def start(self):
        with self._lock:
            if self._stream:
                return
            try:
                self._stream = sd.RawInputStream(
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    dtype=self.dtype,
                    channels=self.channels,
                    callback=self._wrap_callback,
                    device=self.device
                )
                self._stream.start()
                self.logger.info(f"Audio stream started. device={self.device!r}")
            except Exception as e:
                self._stream = None
                self.logger.exception(f"Failed to start audio stream (device={self.device!r}): {e}")
                raise

    def stop(self):
        with self._lock:
            if not self._stream:
                return
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                self.logger.exception("Error while stopping audio stream.")
            finally:
                self._stream = None
                self.logger.info("Audio stream stopped.")

    def is_active(self):
        with self._lock:
            return self._stream is not None
