# Внешний CLI-транскрибер

## Назначение

`python cli.py external-transcribe ...` — стабильная точка входа CLI «один вызов — один файл» для локальных приложений, которым нужно передать AskVLM один медиафайл и получить текст транскрипта.

Поток заточен под машинное взаимодействие:

- Модель Whisper по умолчанию — `small`.
- Провайдер STT по умолчанию — `whisper` (`--stt-provider whisper`). Опционально доступны GigaAM:
  - `gigaam-ctc` — `ai-sage/GigaAM-Multilingual` revision `ctc`, без пунктуации и капитализации; только CPU;
  - `gigaam-e2e-ctc` — `ai-sage/GigaAM-v3` revision `e2e_ctc`, **с** пунктуацией и капитализацией (BPE + ITN); CPU или CUDA.
- Команда по умолчанию пишет только текст транскрипта в `stdout`.
- Диаризация спикеров и LLM-форматирование диалога по умолчанию выключены.
- Модель подгружается по требованию и выгружается после завершения команды (в legacy `--no-daemon`; в режиме демона — резидентно).
- Если CUDA доступна, но выделение VRAM для Whisper или `gigaam-e2e-ctc` не удаётся, AskVLM автоматически продолжает на CPU (подробности — в разделе «Поведение по умолчанию»).
- `gigaam-ctc` не использует CUDA, CUDA→CPU recovery и Windows GPU child-isolation: только CPU.
- На Windows при `--device` отличном от `cpu` (Whisper и `gigaam-e2e-ctc`) команда использует изолированный child-процесс, чтобы снизить влияние upstream crash-on-exit (`faster-whisper`/`ctranslate2`, CUDA-рантайм torch) на вызывающий процесс (только legacy-режим `--no-daemon`).

## Демон-оркестратор и файловая очередь (режим по умолчанию)

Начиная с этой версии, `external-transcribe` по умолчанию работает как **тонкий клиент** единого демона-оркестратора, а не загружает модель в своём процессе:

1. Клиент проверяет, жив ли демон (по heartbeat). Если нет — запускает один **detached** процесс `external-transcribe-daemon` и ждёт его готовности.
2. Клиент кладёт задание в файловую очередь (по умолчанию `<project>/.cache/external_queue`), ждёт результат до `--client-timeout`, печатает транскрипт и завершается.
3. Если истёк `--client-timeout`, клиент **сигнализирует демону дроп** задания (cancel-маркер), чтобы демон не тратил ресурсы на брошенную задачу и очередь не копила «висяки».
4. **Bounded CPU-recovery.** Если запрос был на GPU (`--device` ≠ `cpu`) у провайдера, умеющего CUDA (Whisper или `gigaam-e2e-ctc`), а демон оказался деградирован (истёк `--client-timeout` или демон недоступен), клиент перед выходом делает один встроенный (in-process) проход транскрипции на CPU тем же провайдером. При успехе печатается транскрипт и команда завершается с кодом `0`; только если CPU-проход тоже не дал результата, команда возвращает деградированный код выхода (`10` для таймаута, `11` для недоступного демона). Так зависший/недоступный GPU-демон не теряет транскрипт молча. Для явного `--device cpu` и для `gigaam-ctc` дополнительный проход не выполняется (восстанавливаться не из чего). CPU-проход `gigaam-e2e-ctc` загружает модель в процесс клиента (~2.9 ГБ RAM на время прохода); вызывающим с жёстким бюджетом RAM подходит явный `--device cpu`.

Зачем это нужно:

- **Модель грузится один раз** в резидентном демоне, а не на каждый файл — нет повторного чтения весов с диска и нет «cold-load на сообщение».
- **Один процесс на машину** обслуживает все вызовы (`external-transcribe-daemon` — singleton через lock + heartbeat). Параллелизм ограничен `--workers` (по умолчанию `1`, по доктрине «одна активная нейросеть»).
- **Нет осиротевших воркеров**: тяжёлая работа живёт в демоне, а клиент короткоживущий; его аварийное завершение не оставляет фоновых GPU-процессов.
- Демон сам выключается после простоя (`--idle-shutdown`, по умолчанию 600 с), освобождая VRAM, и просыпается по новому заданию.

Запустить демон вручную (необязательно — клиент поднимает его сам):

```powershell
python cli.py external-transcribe-daemon --workers 1 --whisper-model small --device cuda
```

GigaAM (зависимости уже в `.[ml]`; `gigaam-ctc` — CPU, `gigaam-e2e-ctc` — CPU или CUDA):

```powershell
python cli.py external-transcribe-daemon --stt-provider gigaam-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc
python cli.py external-transcribe-daemon --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe-daemon --stt-provider gigaam-e2e-ctc --device cuda
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cuda
```

Если уже жив демон с другим `--stt-provider`, клиент **не** отправит задание в «чужую» резидентную модель: вернётся unavailable / mismatch. Singleton сохраняется — перезапустите демон с нужным провайдером. Для GPU-запроса провайдера с CUDA (Whisper, `gigaam-e2e-ctc`) такой unavailable запускает bounded CPU-recovery (шаг 4), и транскрипт выдаёт встроенный CPU-проход; с `--device cpu` и для `gigaam-ctc` команда завершается кодом `11`.

Устройство живого демона сверяется с запросом только для диагностики: если heartbeat демона и `--device` клиента расходятся по классу (`cpu` против `auto`/`cuda`), клиент пишет в `stderr` предупреждение с маркером `ASKVLM_DAEMON_DEVICE_MISMATCH`, а задание всё равно обслуживает резидентный демон на своём устройстве. Чтобы сменить устройство, перезапустите демон с нужным `--device`. Heartbeat несёт запрошенный при старте демона токен (`auto`/`cpu`/`cuda`), а не фактическое устройство после выбора и fallback.

Legacy одноразовый запуск без демона (старое поведение «модель в этом же процессе»):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --no-daemon
```

## Установка

Сначала установите проект и ML-зависимости (Whisper + оба GigaAM-провайдера в одном extra):

```powershell
pip install -e .
pip install -e .[ml]
# * run.ps1/build.ps1 по умолчанию чинят CUDA torch 2.10 (после pip с PyPI часто остаётся CPU)
.\run.ps1 -SkipLaunch -Fast
```

Стек `[ml]` включает torch/torchaudio **2.10** (CUDA-колёса чинит `run.ps1`/`build.ps1` по умолчанию: cu128 → cu126), transformers 5, hydra-core, omegaconf, **sentencepiece**, **pyannote.audio**. Последние два нужны remote-code modeling GigaAM даже для short-form `.transcribe` (без longform/VAD). На CPU `gigaam-ctc` занимает порядка ~2.5 ГБ RAM, `gigaam-e2e-ctc` — ~2.9 ГБ RAM (против компактного Whisper Small CPU); на CPU оба не занимают VRAM. `gigaam-e2e-ctc` на CUDA занимает около 1.5–1.65 ГБ VRAM на процесс (CUDA-контекст, веса fp32 ~0.85 ГБ, активации окна 20 с под fp16 autocast; замер nvidia-smi на RTX 3070, torch 2.10 + cu128).

Загрузка модели идёт через Hugging Face Transformers с `trust_remote_code=True` **только** для двух закреплённых репозиториев: `ai-sage/GigaAM-Multilingual` на revision `ctc` и `ai-sage/GigaAM-v3` на revision `e2e_ctc` (официальный remote-code API карточек). Это доверие относится к коду этих моделей/ревизий, а не к произвольным HF-репозиториям. Для v3 обёртка на время `from_pretrained` патчит `hydra.utils.instantiate` (CPU device) и `PreTrainedModel._finalize_model_loading` (compat с transformers 5.x) и **всегда** восстанавливает оригиналы в `finally`, в том числе при ошибке загрузки. Веса v3 всегда загружаются на CPU и приводятся к fp32; для `--device auto|cuda` готовая модель затем целиком переносится на CUDA уже после снятия патчей. Модель целиком в fp16 не переводится: mel-фронтенд (n_fft=320) остаётся fp32, а энкодер на CUDA выполняется под собственным `torch.autocast(float16)` remote-кода. Если remote-code поднимает `ImportError`/`ModuleNotFoundError` из‑за незакрытого `[ml]`, обёртка подсказывает `pip install -e ".[ml]"` и сохраняет имя недостающего модуля.

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

Выбрать GigaAM:

```powershell
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cuda
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

Для `--stt-provider gigaam-ctc` устройство всегда CPU (`auto` → `cpu`); `cuda` отклоняется до загрузки модели.

Для `--stt-provider gigaam-e2e-ctc` токен `--device` (`auto`/`cpu`/`cuda`) проверяется без импорта torch, а разрешается при загрузке модели:

1. `cpu` — модель остаётся на CPU, CUDA не трогается.
2. `cuda` — модель переносится на CUDA, если CUDA доступна; иначе — CPU с предупреждением в логе (команда не падает).
3. `auto` — CUDA выбирается, только если CUDA доступна и `torch.cuda.mem_get_info` показывает не меньше 2.5 ГиБ свободной VRAM (`GIGAAM_AUTO_CUDA_MIN_FREE_VRAM_BYTES` в `core/gigaam_device.py`: пик процесса ~1.6 ГиБ плюс запас для соседа по GPU, например резидентного Whisper-демона); иначе — CPU с предупреждением. Запрос свободной VRAM создаёт CUDA-контекст процесса; если `auto` затем остаётся на CPU, контекст (несколько сотен МиБ) удерживается до выхода процесса.
4. Нехватка памяти GPU при переносе модели возвращает всю модель на CPU (без смешанного размещения весов) и освобождает кэш CUDA.
5. Нехватка памяти GPU на очередном 20-секундном окне переносит модель на CPU, освобождает кэш CUDA и повторяет это же окно на CPU; последующие окна и задания резидентного демона остаются на CPU до перезапуска процесса.
6. Выгрузка модели (`close`/`unload`) после работы на CUDA освобождает кэш CUDA (`torch.cuda.empty_cache`).

На CUDA энкодер работает под fp16 autocast, поэтому текст может незначительно отличаться от CPU-прогона (на 5-минутном тестовом файле — 2 слова из 453). Резидентный демон на CUDA обрабатывает 5-минутный файл примерно в 3.5 раза быстрее, чем на CPU (замер на RTX 3070: около 2.9 с против 10 с на задание с учётом старта клиента); в одноразовом `--no-daemon` на коротких файлах CUDA медленнее CPU из-за инициализации CUDA и child-процесса.

Параметры Whisper (`--compute-type`, beam/VAD) в GigaAM не передаются.

Так перегруженный GPU не блокирует локальную интеграцию, пока в системе ещё достаточно RAM.

Гарантия fallback относится к стандартному одноразовому пути Whisper и к `gigaam-e2e-ctc`. Опции вроде диаризации могут потребовать дополнительной ёмкости GPU. `gigaam-ctc` не участвует в CUDA→CPU recovery.

Помимо OOM-fallback внутри пути модели, режим демона добавляет внешний bounded CPU-fallback на уровне клиента: при таймауте или недоступности GPU-демона выполняется один встроенный проход на CPU (см. шаг 4 выше). Это покрывает случаи, когда GPU-демон завис или аварийно завершился и сам выгрузиться/переключиться на CPU уже не может.

## Надёжность на Windows (subprocess isolation)

Для `external-transcribe` на Windows при `--device != cpu` (Whisper и `gigaam-e2e-ctc`; `gigaam-ctc` всегда работает в процессе на CPU) AskVLM выполняет транскрипцию в child-процессе:

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

## Длинные медиа (GigaAM CTC)

Оба GigaAM-провайдера (`gigaam-ctc` = Multilingual revision `ctc` и `gigaam-e2e-ctc` = v3 revision `e2e_ctc`) отклоняют файлы строго длиннее 400000 сэмплов (~25.0 с при 16 kHz) с `ValueError("Too long wav file, use 'transcribe_longform' method.")`. AskVLM не использует native `transcribe_longform` (pyannote VAD, нужен `HF_TOKEN`). Оконный цикл 20 с одинаков для обеих голов.

Вместо duration-гейта и short-form ветки обёртка **всегда** читает WAV потоково окнами по 20 секунд (в сэмплах исходного файла):

- первое окно короче 20 с (EOF) — один вызов `model.transcribe` по исходному пути;
- иначе каждое окно пишется во временный `chunk_NNN.wav`, транскрибируется отдельно, тексты склеиваются пробелом; сегменты несут абсолютные `start`/`end` (кадровый офсет / sample rate);
- файл ровно 20.0 с может пройти через chunked-путь (один temp-чанк);
- нестандартный WAV-заголовок (`wave.Error` / `EOFError`, включая WAVE_FORMAT_EXTENSIBLE на Python 3.11) — один ре-энкод через ffmpeg в PCM 16 kHz mono, затем тот же оконный цикл.

Отмена длинной джобы кооперативная: pipeline бросает `CancelledError` из `on_segment` между чанками, а не внутри одного вызова модели.

Для `gigaam-e2e-ctc` на CUDA окно, упавшее по нехватке VRAM, повторяется на CPU, поэтому в склейке нет пропущенных окон.

### `--client-timeout` для длинных файлов

В режиме демона дедлайн джобы равен `submitted_at + client_timeout` (по умолчанию 300 с). По истечении клиент ставит cancel-маркер; демон прерывает работу на границе 20-секундного окна.

Вызывающим нужно масштабировать `--client-timeout` с длительностью медиа. Для файлов длиннее нескольких минут значение по умолчанию может оказаться недостаточным: часы аудио на CPU GigaAM не уложатся в 300 с. После таймаута отмена срабатывает кооперативно между окнами, а не мгновенно внутри `model.transcribe`.

## Примечания

- AskVLM хранит кэши моделей в каталоге `.cache/` проекта.
- `external-transcribe` рассчитан на один файл на один запуск процесса.
- `external-extract-frames` рассчитан на один видеофайл на один запуск процесса; кадры остаются в `--output-dir` после завершения.
- Для пакетного экспорта в `txt`, `srt`, `vtt` или `json` используйте `python cli.py transcribe ...`.
