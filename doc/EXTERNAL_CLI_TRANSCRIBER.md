# Внешний CLI-транскрибер

## Назначение

`python cli.py external-transcribe ...` — стабильная точка входа CLI «один вызов — один файл» для локальных приложений, которым нужно передать AskVLM один медиафайл и получить текст транскрипта.

Поток заточен под машинное взаимодействие:

- Модель Whisper по умолчанию — `small`.
- Команда по умолчанию пишет только текст транскрипта в `stdout`.
- Диаризация спикеров и LLM-форматирование диалога по умолчанию выключены.
- Whisper подгружается по требованию и выгружается после завершения команды.
- Если CUDA доступна, но выделение VRAM для Whisper не удаётся, AskVLM автоматически повторяет запуск Whisper на CPU.
- На Windows при `--device` отличном от `cpu` команда использует изолированный child-процесс, чтобы снизить влияние upstream crash-on-exit (`faster-whisper`/`ctranslate2`) на вызывающий процесс.

## Установка

Сначала установите проект и ML-зависимости:

```powershell
pip install -e .
pip install -e .[ml]
```

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

- `--whisper-model small`
- `--device auto`
- `--compute-type auto`
- `--no-diarization`
- `--no-dialog-blocks`
- `--stdout`

Поведение `--device auto`:

1. AskVLM сначала пробует CUDA, если она доступна.
2. Если загрузка модели или инференс на GPU падают из‑за нехватки памяти GPU, AskVLM выгружает Whisper и повторяет запуск на CPU.
3. Fallback на CPU сам подбирает безопасный для CPU тип вычислений.

Так перегруженный GPU не блокирует локальную интеграцию, пока в системе ещё достаточно RAM.

Гарантия fallback относится к стандартному одноразовому пути Whisper. Опции вроде диаризации могут потребовать дополнительной ёмкости GPU.

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
- `--frame-budget INT` (по умолчанию `20`) — максимальное число кадров. `0` отключает ограничение.

Поведение адаптивного FPS:

1. Оценивается `ceil(duration_s × fps)`.
2. Если оценка ≤ `--frame-budget`, используется `--fps`.
3. Если оценка > `--frame-budget`, используется `--fps-fallback`.

Выход:

- По умолчанию — пути к файлам кадров в `stdout`, один на строку.
- С `--json` — JSON-объект `{"frames": [...], "fps_used": N, "duration_s": N}`.
- Код завершения `0` при успехе (в том числе при нулевой длительности видео). Код `1` при ошибке.

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
