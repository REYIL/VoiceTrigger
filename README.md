![Python Versions (3.8, 3.9, 3.10, 3.11, 3.12)](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)
![Vosk](https://img.shields.io/badge/Vosk-blue)

# VoiceTrigger

Realtime-распознаватель речи на базе **[Vosk](https://github.com/alphacep/vosk-api)**: управление микрофоном, детекция уровня голоса (whisper / normal / shout), опциональное шумоподавление, декораторный API для обработки текста, ключевых слов и быстрых команд.

---

## Содержание

1. [Установка](#установка)
2. [Пример: простой голосовой помощник](#пример-простой-голосовой-помощник)
3. [Управление (методы и рекомендации)](#управление-методы-и-рекомендации)
4. [Декораторы и `Filter` (API событий)](#декораторы-и-filter-api-событий)
5. [Калибровка голоса (calibrate_voice.py)](#калибровка-голоса--calibrate_voicepy)
6. [Конфиг для mode — как создать и применить](#конфиг-для-mode--как-создать-и-применить)
7. [Выбор микрофона вручную](#выбор-микрофона-вручную)
8. [Режим шумоподавления (опционально)](#режим-шумоподавления-опционально)
9. [Отладка, советы и частые ошибки](#отладка-советы-и-частые-ошибки)
10. [Структура проекта](#структура-проекта)
11. [Лицензия](#лицензия)

---

## Установка

**Через репозиторий**
1. Склонируйте/скопируйте проект в папку.
2. Установите зависимости:

```bash
pip install -r requirements.txt
```

**Через pip**
```bash
pip install VoiceTrigger
```

> Также скачайте Vosk-модель (например, `model_small`) и укажите путь в `model_path`.

---

## Пример: простой голосовой помощник

Пример использования `VoiceTrigger`. В этом примере помощник «просыпается» по ключевому слову `Алиса`, слушает фразы, реагирует на быстрые команды (`quick_words`), и по длительной тишине возвращается в режим прослушивания ключевых слов.

```python
import asyncio
import time
from pathlib import Path
from VoiceTrigger import (
    VoiceTrigger,
    Filter, TextContext,
    ColorLogger, Mode
)

rms_thresholds = {
    "whisper": -43.0,
    "normal": -15.0,
    "shout": 0.0
}

bot = VoiceTrigger(
    model_path="model_small",  # Путь к модели
    keywords=["Алиса"],  # Не обязательно, может брать автоматически с Filter
    quick_words=["стоп", "назад", "вперед"],  # Не обязательно, может брать автоматически с Filter
    # calibration_path=Path("voice_calibration.json"),
        # Путь указывать не обязательно
        # Если не указывать будет пытаться брать из "voice_calibration.json", а если файла не будет то определение будет работать по системным параметрам
    # rms_thresholds=rms_thresholds,
        # Если калибровка работает плохо, можно сделать ручную настройку
    device=None,  # Устройство ввода, если не указывать выберет сам
    logger=ColorLogger(level="debug")  # Логгер
)

state = {"active_until": 0.0}


@bot.keyword(Filter("Алиса"))
async def on_alisa(ctx: TextContext):
    bot.log.info(f"[KW] {ctx.match} mode={ctx.mode}")
    bot.start_recognition_main()
    bot.stop_recognition_keywords()
    state["active_until"] = time.time() + 10.0


@bot.quick(Filter(["стоп", "пауза"]))
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


@bot.text(Filter(["привет", "здарова"], lv=10, mode=Mode.normal))
async def on_greeting(ctx: TextContext):
    bot.log.info(f"[GREETING] {ctx.match} text='{ctx.text}' mode={ctx.mode}")


@bot.on_silence()  # Возвращает время с последнего quick_words
async def handle_silence_main(sec: float):
    now = time.time()
    if 0 < state["active_until"] <= now and bot.active_main and sec >= 10.0:
        bot.log.info(f"[Silence] {sec:.1f}s -> back to keywords")
        bot.stop_recognition_main()
        bot.start_recognition_keywords()
        state["active_until"] = 0.0


# @bot.on_kw_silence()  # Возвращает время с последнего keywords
# async def handle_kw_silence(sec: float):
#     if sec >= 5.0:
#         bot.log.debug(f"[KW Silence] {sec:.1f}s with no keywords")


if __name__ == "__main__":
    devices = bot.list_input_devices()  # Вывод всех аудио устройств
    bot.log.debug(f"Available input devices: {devices}")
    try:
        asyncio.run(bot.run(initial_keywords_mode=True))  # Запуск с вначале включенным keywords_mode
    except KeyboardInterrupt:
        bot.log.info("Interrupted by user.")
```

---

## Управление (методы и рекомендации)

**Основные методы**:

* `start_recognition_main()` — включить основной режим распознавания (continuous).
* `stop_recognition_main()` — выключить основной режим.
* `start_recognition_keywords()` — включить режим прослушивания ключевых слов.
* `stop_recognition_keywords()` — выключить режим ключевых слов.
* `reload_model(new_model_path=None)` — перезагрузить Vosk-модель (опционально указать новый путь).
* `list_input_devices()` — вернуть список доступных входных устройств (index, name, max_input_channels).
* `set_input_device(device, restart_stream=False)` — установить устройство ввода (индекс или имя). `restart_stream=True` попытается перезапустить поток.

**Рекомендации**:

* **Не рекомендуется** включать одновременно `main` и `keywords`. Эти режимы имеют разные цели — `keywords` оптимизирован для обнаружения wake-word, `main` — для непрерывной речи.
* Если нужна быстрая реакция на короткие команды, используйте **`quick_words` совместно с `main`** — quick-обработчики работают параллельно с основным распознаванием и оптимизированы под короткие команды.
* `keyword`-режим хорош для «пробуждения» (wake word). Обычно вы запускаете `keywords` по умолчанию, а при обнаружении wake-word временно переключаетесь в `main`.

---

## Декораторы и `Filter` (API событий)

**Декораторы**:

* `@bot.text(FILTER?)` — обработчики общего текста (по умолчанию wildcard). Аргумент — `TextContext`.
* `@bot.keyword(FILTER?)` — обработчики ключевых слов; указанные фразы добавляются в список `keywords`.
* `@bot.quick(FILTER?)` — быстрые команды (короткие слова/фразы).
* `@bot.on_silence()` — обработчики тишины для `main` (параметр — количество секунд молчания).
* `@bot.on_kw_silence()` — обработчики тишины для `keywords`.

**Filter**:

```python
Filter(phrases=None | "слово" | ["а","б"], lv=10, mode=Mode.normal|whisper|shout)
```

* `phrases` — список фраз; пустой список / `None` → wildcard (обработчик принимает все тексты).
* `lv` — процент допуска ошибок для Levenshtein (число 0..100). Чем больше — тем сильнее допускаются отличия при сравнении.
* `mode` — (`Mode.whisper`, `Mode.normal`, `Mode.shout`) — если указан, обработчик вызовется только при совпадении голосового режима.

**Контекст обработчика (`TextContext`)**:

* `text` — распознанный текст (final/partial).
* `mode` — строка: `"whisper" | "normal" | "shout"`.
* `match` — совпавшая фраза из `Filter` или `None` (для wildcard).
* `timestamp` — время события (epoch).

---

## Калибровка голоса

В проекте есть модуль `VoiceCalibrator`, она собирает статистику по трём уровням речи (`quiet`, `normal`, `loud`) и сохраняет `voice_calibration.json`. Этот файл используется `VoiceTrigger` для адаптивной настройки порогов RMS/HF и порога тишины.

**Запуск калибровки:**

```python
from pathlib import Path

from VoiceTrigger import VoiceCalibrator

CALIBRATION_PATH = Path(__file__).parent / "voice_calibration.json"

VoiceCalibrator.calibrate(calibration_path=CALIBRATION_PATH)  # Путь указывать не обязательно
```

Скрипт попросит записать несколько фрагментов для каждого уровня и сохранит средние значения в `voice_calibration.json`.

**Почему калибровка полезна:**

* Подстраивает пороги под конкретный микрофон, акустику комнаты и расстояние до источника.
* Повышает корректность определения whisper/normal/shout.

**Минусы калибровки:**

* Требует аккуратного прохождения процедуры пользователем — если человек говорит слишком громко или тихо, результаты могут быть некорректными.
* Чувствительна к положению микрофона: смена расстояния или угла может сильно изменить RMS и HF, что исказит пороги.
* Автопороги могут оказаться слишком близко друг к другу, особенно если различие между whisper, normal и shout маленькое — классификация становится менее надёжной.
* Плохие условия записи (шум, эхо, фоновые источники) могут «засорить» калибровку.
* При многократной смене среды/оборудования нужна новая калибровка, иначе точность падает.

---

## Конфиг для `mode` — как создать и применить

Можно задать пороги вручную через JSON-конфиг, либо использовать `voice_calibration.json`, полученный через `calibrate_voice.py`.

**Пример `mode_config.json`:**

```json
{
  "rms_thresholds": {
    "whisper": -45.0,
    "normal": -18.0,
    "shout": -1.0
  },
  "hf_ratio_threshold": 1.5,
  "silence_db": -46.0
}
```

**Применение конфигурации в коде:**

```python
import json
from pathlib import Path
from VoiceTrigger import VoiceTrigger

cfg = json.loads(Path("mode_config.json").read_text(encoding="utf-8"))
bot = VoiceTrigger(...)

bot.voice_detector.rms_thresholds = cfg.get("rms_thresholds", bot.voice_detector.rms_thresholds)
bot.voice_detector.hf_ratio_threshold = cfg.get("hf_ratio_threshold", bot.voice_detector.hf_ratio_threshold)
bot.voice_detector.silence_db = cfg.get("silence_db", bot.voice_detector.silence_db)
```

Или положите результаты калибровки в `voice_calibration.json` — `VoiceTrigger` автоматически прочитает его при создании `VoiceTrigger` (если файл доступен).

**А также можно использовать `rms_thresholds`**

```python
from VoiceTrigger import VoiceTrigger, ColorLogger

rms_thresholds = {
    "whisper": -43.0,
    "normal": -15.0,
    "shout": 0.0
}

bot = VoiceTrigger(
    model_path="model_small",
    quick_words=["стоп", "назад", "вперед", "какая погода"],
    logger=ColorLogger(level="debug"),
    rms_thresholds=rms_thresholds
)
```

---

## Выбор микрофона вручную

**Список устройств:**

```python
devices = VoiceTrigger.list_input_devices()
# или через экземпляр:
devices = bot.list_input_devices()
```

**Установка устройства:**

* При инициализации:

```python
bot = VoiceTrigger(..., device=2)  # индекс
# или
bot = VoiceTrigger(..., device="USB Microphone")  # имя устройства
```

* Во время работы:

```python
bot.set_input_device(2, restart_stream=True)
```

`restart_stream=True` попытается перезапустить поток (может потребоваться освобождение устройства системой).

---

## Режим шумоподавления (опционально)

Можно включить встроенное шумоподавление для микрофона. Для этого необходимо:

1. Установить зависимости:

   ```bash
   pip install noisereduce scipy
   ```
2. Включить режим в коде:

   ```python
   bot = VoiceTrigger(..., noise_reduction=True)
   ```

*Когда использовать:*

* если в помещении много фонового шума (ПК-вентиляторы, улица, кондиционер),
* при записи на встроенные микрофоны ноутбука,
* если нужно повысить точность распознавания.

---

## Отладка, советы и частые ошибки

* Включите подробный логгер:

```python
logger = ColorLogger(level="debug")
bot = VoiceTrigger(..., logger=logger)
```

* Если модель не загружается — проверьте `model_path` и файлы модели.
* Если нет звука или пустые результаты — проверьте `sounddevice.query_devices()` и системные права доступа к микрофону.
* Если плохое распознавание:

  * проверьте sample rate (обычно 16000),
  * расположение микрофона,
  * при необходимости запустите `calibrate_voice.py`,
  * попробуйте включить `noise_reduction` (если установлен `noisereduce`).
* Для коротких команд используйте `quick_words` (работают быстрее и подходят для single-word commands).

---

## Структура проекта

```
.
├── model_small # vosk
├── voicetrigger
│   ├── core
│   │   ├── asmanager.py
│   │   ├── decorators.py
│   │   ├── speechr.py
│   │   └── vldetector.py
│   ├── services
│   │   └── calibration.py
│   ├── utils
│   │   ├── filter.py
│   │   ├── levenshtein.py
│   │   └── logger.py
│   └── __init__.py
├── main.py
└── requirements.txt
```
---

## Лицензия

MIT License — свободно используйте и модифицируйте.
