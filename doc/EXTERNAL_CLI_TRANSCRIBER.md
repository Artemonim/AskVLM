# Внешний CLI-транскрибер

## Назначение

`python cli.py external-transcribe ...` — стабильная точка входа CLI «один вызов — один файл» для локальных приложений, которым нужно передать AskVLM один медиафайл и получить текст транскрипта.

Поток заточен под машинное взаимодействие:

- Модель Whisper по умолчанию — `small`.
- Команда по умолчанию пишет только текст транскрипта в `stdout`.
- Диаризация спикеров и LLM-форматирование диалога по умолчанию выключены.
- Whisper подгружается по требованию и выгружается после завершения команды.
- Если CUDA доступна, но выделение VRAM для Whisper не удаётся, AskVLM автоматически повторяет запуск Whisper на CPU.

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

- `0` — транскрипция завершилась успешно.
- Ненулевой код — AskVLM не смог обработать файл.

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

## Примечания

- AskVLM хранит кэши моделей в каталоге `.cache/` проекта.
- `external-transcribe` рассчитан на один файл на один запуск процесса.
- Для пакетного экспорта в `txt`, `srt`, `vtt` или `json` используйте `python cli.py transcribe ...`.
