# Внешний CLI-транскрибер

## Назначение

`python cli.py external-transcribe ...` — стабильная точка входа CLI «один вызов — один файл» для локальных приложений, которым нужно передать AskVLM один медиафайл и получить текст транскрипта.

Поток заточен под машинное взаимодействие:

- Модель Whisper по умолчанию — `small`.
- Провайдер STT по умолчанию — `whisper` (`--stt-provider whisper`). Опционально доступен CPU-only `gigaam-ctc` (GigaAM Multilingual, revision `ctc`).
- Команда по умолчанию пишет только текст транскрипта в `stdout`.
- Диаризация спикеров и LLM-форматирование диалога по умолчанию выключены.
- Модель подгружается по требованию и выгружается после завершения команды (в legacy `--no-daemon`; в режиме демона — резидентно).
- Если CUDA доступна, но выделение VRAM для Whisper не удаётся, AskVLM автоматически повторяет запуск Whisper на CPU.
- GigaAM CTC не использует CUDA, CUDA-fallback Whisper и Windows GPU child-isolation: только CPU.
- На Windows при `--device` отличном от `cpu` (только Whisper) команда использует изолированный child-процесс, чтобы снизить влияние upstream crash-on-exit (`faster-whisper`/`ctranslate2`) на вызывающий процесс (только legacy-режим `--no-daemon`).

## Демон-оркестратор и файловая очередь (режим по умолчанию)

Начиная с этой версии, `external-transcribe` по умолчанию работает как **тонкий клиент** единого демона-оркестратора, а не загружает модель в своём процессе:

1. Клиент проверяет, жив ли демон (по heartbeat). Если нет — запускает один **detached** процесс `external-transcribe-daemon` и ждёт его готовности.
2. Клиент кладёт задание в файловую очередь (по умолчанию `<project>/.cache/external_queue`), ждёт результат до `--client-timeout`, печатает транскрипт и завершается.
3. Если истёк `--client-timeout`, клиент **сигнализирует демону дроп** задания (cancel-маркер), чтобы демон не тратил ресурсы на брошенную задачу и очередь не копила «висяки».
4. **Bounded CPU-recovery.** Если запрос был на GPU (`--device` ≠ `cpu`), а демон оказался деградирован (истёк `--client-timeout` или демон недоступен), клиент перед выходом делает один встроенный (in-process) проход транскрипции на CPU. При успехе печатается транскрипт и команда завершается с кодом `0`; только если CPU-проход тоже не дал результата, команда возвращает деградированный код выхода (`10` для таймаута, `11` для недоступного демона). Так зависший/недоступный GPU-демон больше не теряет транскрипт молча. Для явного `--device cpu` дополнительный проход не выполняется (восстанавливаться не из чего).

Зачем это нужно:

- **Модель грузится один раз** в резидентном демоне, а не на каждый файл — нет повторного чтения весов с диска и нет «cold-load на сообщение».
- **Один процесс на машину** обслуживает все вызовы (`external-transcribe-daemon` — singleton через lock + heartbeat). Параллелизм ограничен `--workers` (по умолчанию `1`, по доктрине «одна активная нейросеть»).
- **Нет осиротевших воркеров**: тяжёлая работа живёт в демоне, а клиент короткоживущий; его аварийное завершение не оставляет фоновых GPU-процессов.
- Демон сам выключается после простоя (`--idle-shutdown`, по умолчанию 600 с), освобождая VRAM, и просыпается по новому заданию.

Запустить демон вручную (необязательно — клиент поднимает его сам):

```powershell
python cli.py external-transcribe-daemon --workers 1 --whisper-model small --device cuda
```

GigaAM CTC (CPU; зависимости уже в `.[ml]`):

```powershell
python cli.py external-transcribe-daemon --stt-provider gigaam-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc
```

Если уже жив демон с другим `--stt-provider`, клиент **не** отправит задание в «чужую» резидентную модель: вернётся unavailable / mismatch. Singleton сохраняется — перезапустите демон с нужным провайдером.

Legacy одноразовый запуск без демона (старое поведение «модель в этом же процессе»):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --no-daemon
```

## Установка

Сначала установите проект и ML-зависимости (Whisper + GigaAM CTC в одном extra):

```powershell
pip install -e .
pip install -e .[ml]
# * run.ps1/build.ps1 по умолчанию чинят CUDA torch 2.10 (после pip с PyPI часто остаётся CPU)
.\run.ps1 -SkipLaunch -Fast
```

Стек `[ml]` включает torch/torchaudio **2.10** (CUDA-колёса чинит `run.ps1`/`build.ps1` по умолчанию: cu128 → cu126), transformers 5, hydra-core, omegaconf, **sentencepiece**, **pyannote.audio**. Последние два нужны remote-code modeling GigaAM даже для short-form `.transcribe` (без longform/VAD). GigaAM на CPU занимает порядка ~2.5 ГБ RAM (против компактного Whisper Small CPU), зато не занимает VRAM и по скорости сопоставим с Whisper Small на GPU (см. бенчмарки вне репозитория).

Загрузка модели идёт через Hugging Face Transformers с `trust_remote_code=True` **только** для репозитория `ai-sage/GigaAM-Multilingual` на revision `ctc` (официальный remote-code API карточки). Это доверие относится к коду этой модели/ревизии, а не к произвольным HF-репозиториям. Если remote-code поднимает `ImportError`/`ModuleNotFoundError` из‑за незакрытого `[ml]`, обёртка подсказывает `pip install -e ".[ml]"` и сохраняет имя недостающего модуля.

Не ставьте `torch` с PyPI «поверх» CUDA-сборки без последующего `run.ps1`/`build.ps1`: голый `pip install -e .[ml]` часто оставляет CPU `torch 2.10.0`. Отказ от авточинки: `-SkipEnsureCUDA`. Подробности — в `doc/CUDA_SETUP.md`.

Если используете виртуальное окружение в Windows:

```powershell
. .\.venv\Scripts\Activate.ps1
```

## Базовое использование

Вернуть текст транскрипта в `stdout`:

```powershell
python cli.py external-transcribe "C:\media\call.wav"
```

Записать транскрипт в файл и оставить вывод в `stdout`:

```powershell
python cli.py external-transcribe "C:\media\call.wav" `
  --output-file "C:\media\call.txt"
```

Записать транскрипт только в файл:

```powershell
python cli.py external-transcribe "C:\media\call.wav" `
  --output-file "C:\media\call.txt" `
  --no-stdout
```

Принудительно CPU:

```powershell
python cli.py external-transcribe "C:\media\call.wav" --device cpu
```

Явно указать язык:

```powershell
python cli.py external-transcribe "C:\media\call.wav" --language ru
```

Выбрать GigaAM CTC (CPU):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc --device cpu
```

## Контракт команды

Команда:

```text
python cli.py external-transcribe INPUT_PATH [options]
```

Вход:

- `INPUT_PATH` — один аудио- или видеофайл.

Выход:

- По умолчанию команда печатает итоговый текст транскрипта в `stdout`.
- Если указан `--output-file`, тот же обычный текст дополнительно записывается в этот файл.
- При `--no-stdout` нужно также передать `--output-file`.

Код выхода:

- `0` — транскрипция завершилась успешно, включая случай пустого или пробельного результата после `str.strip()`.
- `1` — ошибка выполнения без валидного успешного результата.
- Другой ненулевой код — AskVLM не смог обработать файл или сбой на этапе выполнения.

Пустой транскрипт: если `get_full_text()` даёт строку, которая после `strip()` пустая, это считается штатным успешным завершением. Команда:

- не пишет пустую строку в `stdout` (то есть не добавляет лишний перевод строки),
- создаёт `--output-file` как пустой файл, если путь передан,
- не печатает диагностическое сообщение в `stderr`.

## Поведение по умолчанию

Внешний CLI использует эти значения по умолчанию, пока вы их не переопределите:

- `--stt-provider whisper`
- `--whisper-model small`
- `--device auto`
- `--compute-type auto`
- `--no-diarization`
- `--no-dialog-blocks`
- `--stdout`

Поведение `--device auto` для Whisper:

1. AskVLM сначала пробует CUDA, если она доступна.
2. Если загрузка модели или инференс на GPU падают из‑за нехватки памяти GPU, AskVLM выгружает Whisper и повторяет запуск на CPU.
3. Fallback на CPU сам подбирает безопасный для CPU тип вычислений.

Для `--stt-provider gigaam-ctc` устройство всегда CPU (`auto` → `cpu`); `cuda` отклоняется до загрузки модели. Параметры Whisper (`--compute-type`, beam/VAD) в GigaAM не передаются.

Так перегруженный GPU не блокирует локальную интеграцию, пока в системе ещё достаточно RAM.

Гарантия fallback относится к стандартному одноразовому пути Whisper. Опции вроде диаризации могут потребовать дополнительной ёмкости GPU. GigaAM не участвует в CUDA→CPU recovery Whisper.

Помимо OOM-fallback внутри пути Whisper, режим демона добавляет внешний bounded CPU-fallback на уровне клиента: при таймауте или недоступности GPU-демона выполняется один встроенный проход на CPU (см. шаг 4 выше). Это покрывает случаи, когда GPU-демон завис или аварийно завершился и сам выгрузиться/переключиться на CPU уже не может.

## Надёжность на Windows (subprocess isolation)

Для `external-transcribe` на Windows при `--device != cpu` AskVLM выполняет транскрипцию в child-процессе:

1. Parent запускает тот же CLI entrypoint с внутренними скрытыми флагами child-mode.
2. Child пишет служебный JSON-результат до `pipeline.close(...)`.
3. Parent считает этот JSON источником истины:
   - если JSON валиден и содержит `status=ok`, это успех даже при crash-like коде завершения child после записи результата;
   - если валидного успешного JSON нет и код завершения похож на native crash, parent делает ровно один retry на CPU;
   - если валидного успешного JSON нет, ошибка не crash-like и stderr не содержит маркер внутренней IPC-ошибки — parent отдаёт ошибку наружу без CPU retry;
   - если валидного успешного JSON нет и stderr содержит маркер внутренней IPC-ошибки (сбой инициализации child-процесса), parent выполняет один retry на CPU как защитная мера.

## JIT-загрузка и выгрузка

Команда рассчитана на одноразовый запуск:

1. AskVLM разбирает аргументы CLI.
2. AskVLM готовит аудио во временной рабочей папке, если не задан `--work-dir`.
3. Whisper загружается в память только когда начинается транскрипция.
4. AskVLM возвращает транскрипт.
5. Whisper выгружается до завершения процесса.

Если `--work-dir` не указан, AskVLM создаёт временную папку и удаляет её после завершения.

## Рекомендуемый паттерн интеграции

### PowerShell

```powershell
$transcript = python cli.py external-transcribe "C:\media\call.wav"
if ($LASTEXITCODE -ne 0) {
    throw "Ошибка транскрипции AskVLM."
}
Write-Host $transcript
```

### Подпроцесс Python

```python
import subprocess

result = subprocess.run(
    ["python", "cli.py", "external-transcribe", r"C:\media\call.wav"],
    capture_output=True,
    text=True,
    check=False,
)

if result.returncode != 0:
    raise RuntimeError(result.stderr or "Ошибка транскрипции AskVLM")

transcript_text = result.stdout
```

## Дополнительные возможности

Включить диаризацию спикеров:

```powershell
python cli.py external-transcribe "C:\media\meeting.mp3" --diarization
```

Включить LLM-очистку текста:

```powershell
python cli.py external-transcribe "C:\media\draft.wav" --dialog-blocks
```

Эти опции по умолчанию выключены: они увеличивают стоимость старта и могут подгружать дополнительные ML-бэкенды.

## Извлечение кадров из видео (`external-extract-frames`)

`python cli.py external-extract-frames ...` — внешний CLI «один вызов — один видеофайл» для приложений, которым нужно извлечь кадры с адаптивным FPS.

### Контракт

```text
python cli.py external-extract-frames INPUT_PATH --output-dir PATH [options]
```

Входные аргументы:

- `INPUT_PATH` — видеофайл.
- `--output-dir PATH` — каталог для сохранения кадров (создаётся автоматически).

Параметры выборки:

- `--fps FLOAT` (по умолчанию `0.5`) — целевой FPS.
- `--fps-fallback FLOAT` (по умолчанию `0.2`) — FPS при превышении бюджета.
- `--frame-budget INT` (по умолчанию `20`) — жёсткий потолок числа извлекаемых кадров. `0` отключает ограничение.

Поведение адаптивного FPS:

1. Оценивается `ceil(duration_s × fps)`.
2. Если оценка ≤ `--frame-budget`, используется `--fps`.
3. Если оценка > `--frame-budget`, используется `--fps-fallback`.
4. Если и на fallback оценка превышает `--frame-budget`, используется `frame_budget / duration_s` — бюджетное число кадров равномерно покрывает всю длительность.
5. Лишний кадр из-за округления ffmpeg усекается: вывод никогда не превышает `--frame-budget`.

Выход:

- По умолчанию — пути к файлам кадров в `stdout`, один на строку.
- С `--json` — JSON-объект `{"frames": [...], "fps_used": N, "duration_s": N}`.
- Код завершения `0` при успехе (в том числе при нулевой длительности видео). Код `1` при ошибке.

Устойчивость к colorspace:

- Некоторые контейнеры помечают кадры цветовой матрицей, которую libswscale не может привести к RGB для image-энкодера (ffmpeg падает с `Invalid color space`).
- Сначала кадры извлекаются «как есть»; при таком сбое извлечение повторяется с усиливающимися colorspace-нормализующими `-vf` стратегиями (сброс тега в BT.709, форсирование матриц swscale, нормализация pixel format).
- Полная потеря кадров (код `1`) наступает только если все стратегии не дали ни одного кадра.

### Примеры

```powershell
# Извлечь кадры в /tmp/frames с адаптивным FPS
python cli.py external-extract-frames "C:\media\clip.mp4" `
  --output-dir "C:\media\frames"

# Получить JSON-манифест
python cli.py external-extract-frames "C:\media\clip.mp4" `
  --output-dir "C:\media\frames" --json

# Форсировать 1 FPS, без ограничения числа кадров
python cli.py external-extract-frames "C:\media\clip.mp4" `
  --output-dir "C:\media\frames" --fps 1.0 --frame-budget 0
```

### Рекомендуемый паттерн интеграции (Python)

```python
import json
import subprocess

result = subprocess.run(
    [
        "python", "cli.py", "external-extract-frames",
        r"C:\media\clip.mp4",
        "--output-dir", r"C:\media\frames",
        "--json",
    ],
    capture_output=True,
    text=True,
    check=False,
)

if result.returncode != 0:
    raise RuntimeError(result.stderr or "AskVLM frame extraction error")

manifest = json.loads(result.stdout)
frame_paths = manifest["frames"]
fps_used    = manifest["fps_used"]
```

## Примечания

- AskVLM хранит кэши моделей в каталоге `.cache/` проекта.
- `external-transcribe` рассчитан на один файл на один запуск процесса.
- `external-extract-frames` рассчитан на один видеофайл на один запуск процесса; кадры остаются в `--output-dir` после завершения.
- Для пакетного экспорта в `txt`, `srt`, `vtt` или `json` используйте `python cli.py transcribe ...`.
