import asyncio
import time
# from pathlib import Path

from VoiceTrigger import VoiceTrigger, ColorLogger, Filter, Mode, TextContext, VoiceCalibrator

# CALIBRATION_PATH = Path(__file__).parent / "voice_calibration.json"

# VoiceCalibrator.calibrate(calibration_path=CALIBRATION_PATH)

rms_thresholds = {
    "whisper": -43.0,
    "normal": -15.0,
    "shout": 0.0
}

bot = VoiceTrigger(
    model_path="model_small",
    quick_words=["стоп", "назад", "вперед", "какая погода"],
    logger=ColorLogger(level="debug"),
    # calibration_path=CALIBRATION_PATH
    rms_thresholds=rms_thresholds
)


state = {"active_until": 0.0}


@bot.keyword(Filter("Алиса"))
async def on_alisa(ctx: TextContext):
    bot.log.info(f"[KW] {ctx.match} mode={ctx.mode}")
    bot.start_recognition_main()
    bot.stop_recognition_keywords()
    state["active_until"] = time.time() + 10.0


@bot.quick()
async def on_quick(ctx: TextContext):
    bot.log.info(f"[QUICK] {ctx.match} mode={ctx.mode}")
    if ctx.match and ctx.match.lower() == "стоп":
        bot.stop_recognition_main()
        bot.start_recognition_keywords()
        state["active_until"] = 0.0


@bot.text()
async def on_all_text(ctx: TextContext):
    if ctx.match is None and ctx.text:
        bot.log.info(f"[TEXT] mode={ctx.mode} text='{ctx.text}'")


@bot.on_silence()  # Возвращает время с последнего quick_words
async def handle_silence_main(sec: float):
    now = time.time()
    if 0 < state["active_until"] <= now and bot.active_main and sec >= 10.0:
        bot.log.info(f"[Silence] {sec:.1f}s -> back to keywords")
        bot.stop_recognition_main()
        bot.start_recognition_keywords()
        state["active_until"] = 0.0


if __name__ == "__main__":
    try:
        asyncio.run(bot.run(initial_keywords_mode=True))  # Запуск с вначале включенным keywords_mode
    except KeyboardInterrupt:
        bot.log.info("Interrupted by user.")
