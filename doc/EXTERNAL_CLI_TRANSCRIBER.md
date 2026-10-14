# Внешний CLI-транскрибер

## Назначение

`python cli.py external-transcribe ...` — стабильная точка входа CLI для локальных приложений, которые передают AskVLM медиафайлы и получают текст транскрипта. Один вход сохраняет контракт «один вызов — один файл» (простой текст в `stdout`); несколько входов в одном вызове возвращают по одной JSON-строке на вход.

Поток заточен под машинное взаимодействие:

- Модель Whisper по умолчанию — `small`.
- Провайдер STT по умолчанию — `whisper` (`--stt-provider whisper`). Опционально доступны GigaAM:
  - `gigaam-ctc` — `ai-sage/GigaAM-Multilingual` revision `ctc`, без пунктуации и капитализации; только CPU;
  - `gigaam-e2e-ctc` — `ai-sage/GigaAM-v3` revision `e2e_ctc`, **с** пунктуацией и капитализацией (BPE + ITN); CPU или CUDA.
- Команда по умолчанию пишет только текст транскрипта в `stdout`.
- Диаризация спикеров и LLM-форматирование диалога по умолчанию выключены.
- По умолчанию файлы идут через общий STT-супервизор: процессы-воркеры держат модели резидентно, по процессу на конфигурацию (провайдер, модель, тип вычислений, CPU или GPU). В legacy `--no-daemon` модель грузится в процессе самой команды и выгружается перед выходом.
- Через супервизор устройство строгое: `--device cpu` никогда не использует GPU, `--device cuda` остаётся на GPU (на CPU — только если CUDA-устройства нет, с предупреждением); `auto` предпочитает GPU и уходит на CPU, только если там результат ожидается раньше. Внутрипроцессные CUDA→CPU fallback из раздела «Поведение по умолчанию» действуют только в legacy `--no-daemon`.
- `gigaam-ctc` не использует CUDA, CUDA→CPU recovery и Windows GPU child-isolation: только CPU.
- На Windows при `--device` отличном от `cpu` (Whisper и `gigaam-e2e-ctc`) команда использует изолированный child-процесс, чтобы снизить влияние upstream crash-on-exit (`faster-whisper`/`ctranslate2`, CUDA-рантайм torch) на вызывающий процесс (только legacy-режим `--no-daemon`).

## STT-супервизор и пул воркеров (режим по умолчанию)

По умолчанию `external-transcribe` — **тонкий клиент** одного STT-супервизора (`external-transcribe-daemon`), который управляет пулом процессов-воркеров. Сам клиент модель не загружает.

### Ход вызова

1. Клиент заводит один дедлайн по настенным часам на весь вызов (`--client-timeout`, по умолчанию 300 с), узнаёт длительность каждого входа через `ffprobe` (неудача означает лишь «длительность неизвестна») и пишет по заданию на вход в файловую очередь `<корень очереди>/v2/` (корень: `ASKVLM_EXTERNAL_QUEUE_DIR` или `<project>/.cache/external_queue`). У всех входов одного вызова общий id вызова, `--priority` и `--max-parallel`.
2. Если свежего heartbeat супервизора нет, клиент запускает **detached** супервизор (с `CREATE_BREAKAWAY_FROM_JOB`, если job-объект вызывающего это разрешает, чтобы супервизор пережил вызывающего) и перезапускает его не чаще раза в 10 с, пока никто не отвечает. Если за 30 с супервизор так и не стал доступен, незавершённые входы получают `unavailable` (код `11`); в сообщении указан лог супервизора `<корень очереди>/v2/control/supervisor.out.log`.
3. Клиент ждёт результаты до дедлайна. По дедлайну он пишет cancel-маркеры для незавершённых заданий (супервизор снимает или останавливает их) и сообщает `timeout` (код `10`). `--client-timeout inf` ждёт без дедлайна.
4. **Встроенного CPU-recovery в клиенте нет**: GPU-задание, которое не удалось завершить, не перезапускается на CPU внутри процесса клиента.

### Супервизор и воркеры

- **Один супервизор на корень очереди** под файловой блокировкой ОС (`v2/control/supervisor.lock`); второй супервизор сразу завершается с кодом `0`. Heartbeat (`v2/control/supervisor.heartbeat.json`) несёт лимиты, список воркеров и размер очереди.
- **Воркер — это процесс**, который держит резидентно одну модель и обслуживает одну конфигурацию (ключ воркера): провайдер, модель, тип вычислений с нормализацией по бэкенду (Whisper: `int8` на CPU и `float16` на GPU для `auto`; GigaAM: `fp32`) и бэкенд `cpu` или `gpu`. Язык задаётся на задание и в ключ не входит.
  - CPU-воркер: одна модель на несколько одновременных заданий (CPU-слоты); каждый слот работает на `threads_per_slot` потоках (по умолчанию 2).
  - GPU-воркер: один процесс, который батчит задания между собой — до 8 20-секундных окон GigaAM или до 4 клипов Whisper не длиннее 30 с с явным языком (остальные задания Whisper идут последовательно). Инференс GigaAM на CUDA никогда не идёт из нескольких потоков.
- **Воркеры запускаются по спросу** на конфигурации из очереди и останавливаются после простоя (TTL, по умолчанию 600 с, `--idle-shutdown`). Если новой конфигурации не хватает ресурсов, сначала останавливаются простаивающие воркеры других конфигураций (давно не использованные первыми). Занятые воркеры ради ресурсов не останавливаются никогда. Супервизор завершается вскоре после того, как последний воркер остановился при пустой очереди.
- **Каждый воркер живёт в своём Windows Job Object** с лимитом памяти, равным его резерву commit; всё дерево процессов (включая `ffmpeg`) умирает вместе с job, поэтому осиротевших GPU-процессов не остаётся. Воркер, превысивший резерв, падает один; следующий резерв для этой конфигурации растёт на 25%.
- **Допуск по ресурсам:** до старта воркера супервизор резервирует CPU-слоты и потоки, commit (private bytes Windows) и, для GPU-воркеров, VRAM. Одновременно идёт не больше одной холодной загрузки; свободный commit системы и физически свободная VRAM (`nvidia-smi`) сверяются с защитными запасами. Запрос, который никогда не уложится в заданные лимиты, получает `unavailable` с цифрами.
- **Политика устройств:** `cpu` работает на CPU; `cuda` работает на GPU и ждёт GPU-мощности (на CPU уходит, только если CUDA-устройства нет, с предупреждением `ASKVLM_CUDA_UNAVAILABLE`); `auto` предпочитает GPU и переносит задание на CPU, только если оценка времени до результата на CPU меньше 0.8 × оценки ожидания GPU, поэтому длинная очередь `auto` может занять и GPU, и CPU-слоты. `gigaam-ctc` работает только на CPU. Результаты CPU и GPU могут немного различаться.
- **Приоритет и справедливая доля:** `--priority high|normal|low` (по умолчанию `normal`). Старший класс допускается первым; внутри класса планировщик чередует вызовы, поэтому вызов с одним файлом не ждёт целую пачку. Уже начатые задания никогда не вытесняются. Ожидающее задание повышается на класс каждые `priority_aging_s` (по умолчанию 300 с), так что работа `low` со временем обслуживается.
- **Сбои:** падение воркера повторяет его задание один раз (`max_attempts` 2, счётчик переживает перезапуск супервизора); конфигурация, чья модель не загрузилась, уходит в cooldown, и её задания завершаются `unavailable` с текстом ошибки загрузки; текст ошибки воркера (например `Too long wav file`) доходит до `stderr` клиента.
- Выученные времена и пики памяти хранятся в `v2/state/estimates.json`; логи — `v2/control/supervisor.log` и `v2/logs/<worker>.log`.
- Ещё работающий старый (v1) демон сканирует только `<корень очереди>/incoming` и v2-задания не берёт; он завершится после своего простоя.

### Лимиты и конфигурация

Супервизор берёт лимиты из первого найденного источника: `external-transcribe-daemon --config PATH`, переменная окружения `ASKVLM_STT_POOL_CONFIG`, `<project>/stt_pool.json` (в `.gitignore`, свой на каждой машине) или консервативные встроенные значения (2 CPU-слота, 1 GPU-воркер, 12 ГиБ commit, 4 ГиБ VRAM). `stt_pool.example.json` в корне проекта показывает все ключи со значениями для машины на 8 CPU-слотов / 24 ГиБ commit и 2 GPU-воркера / 7 ГиБ VRAM; скопируйте его в `stt_pool.json` и поправьте. Неизвестные ключи и неверные значения останавливают супервизор с ошибкой.

| Ключ | Смысл |
| --- | --- |
| `cpu.max_slots`, `cpu.threads_per_slot`, `cpu.max_slots_per_worker` | Одновременные CPU-задания, потоки на задание, слоты на процесс CPU-воркера |
| `gpu.max_workers`, `gpu.batch_lanes`, `gpu.host_threads`, `gpu.scale_out_backlog_s` | Процессы GPU-воркеров, дорожки батча (GigaAM/Whisper), CPU-потоки, зарезервированные под GPU-воркер, очередь, оправдывающая ещё один GPU-воркер |
| `threads.budget` | Единый бюджет CPU-потоков на CPU-слоты и хостовые потоки GPU-воркеров |
| `memory.commit_budget_gib`, `memory.free_commit_guard_gib`, `memory.critical_free_commit_gib` | Бюджет commit на все воркеры вместе, запас свободного commit для нового воркера, свободный commit, ниже которого простаивающие воркеры останавливаются |
| `vram.budget_gib`, `vram.free_guard_gib` | Бюджет VRAM на все GPU-воркеры, запас свободной VRAM для нового GPU-воркера |
| `lifecycle.*` | TTL простоя воркера, задержка выхода супервизора, сроки остановки и отмены |
| `scheduling.*` | Порог переноса на CPU, защита от голодания, попытки, коэффициент взаимного влияния GPU/CPU, старение приоритета, cooldown |

Запустить супервизор вручную (необязательно — клиент поднимает его сам):

```powershell
python cli.py external-transcribe-daemon
python cli.py external-transcribe-daemon --config C:\path\stt_pool.json --idle-shutdown 300
python cli.py external-transcribe-daemon --cpu-slots 4 --gpu-workers 1 --commit-budget-gb 16 --vram-budget-gb 6
```

Старые флаги демона `--workers`, `--stt-provider`, `--whisper-model`, `--device`, `--compute-type` и `--language` принимаются и игнорируются (в `stderr` идёт пометка `ASKVLM_DEPRECATED`): пул обслуживает любую конфигурацию.

### Пакетные вызовы

```powershell
python cli.py external-transcribe a.wav b.mp3 c.ogg --max-parallel 2 --priority low --output-dir C:\out
```

- При нескольких входах `stdout` несёт по JSON-объекту на вход в порядке входов: `{"index", "input", "status", "text", "error"}` (плюс `"output"` с `--output-dir`); строка пишется, как только все предыдущие входы завершены. `status`: `ok`, `empty`, `error`, `timeout` или `unavailable`.
- `--output-dir DIR` пишет `<stem>.txt` для каждого успешного входа; повторяющиеся имена получают суффиксы `-2`, `-3`, …, которые выбираются до запуска.
- `--output-file` при нескольких входах пишется один раз в конце: транскрипты через перевод строки в порядке входов (неудачный вход даёт пустую строку).
- `--max-parallel N` ограничивает, сколько заданий этого вызова идёт одновременно; на другие вызовы он не влияет. `--daemon-workers N` — устаревший синоним (пометка в `stderr`).
- Код выхода: `10`, если хоть один вход упёрся в таймаут, иначе `11`, если хоть один недоступен, иначе `1`, если хоть один завершился ошибкой, иначе `0`.

`--diarization`, `--dialog-blocks` и `--work-dir` через супервизор не поддерживаются: команда завершается кодом `2` и подсказывает `--no-daemon`.

Примеры GigaAM (зависимости в `.[ml]`; `gigaam-ctc` — CPU, `gigaam-e2e-ctc` — CPU или CUDA):

```powershell
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-ctc
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cpu
python cli.py external-transcribe "C:\media\call.wav" --stt-provider gigaam-e2e-ctc --device cuda
```

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
python cli.py external-transcribe INPUT_PATH [INPUT_PATH ...] [options]
```

Вход:

- `INPUT_PATH` — аудио- или видеофайл; через супервизор можно передать несколько файлов (`--no-daemon` принимает ровно один).

Выход (один вход):

- По умолчанию команда печатает итоговый текст транскрипта в `stdout`.
- Если указан `--output-file`, тот же обычный текст дополнительно записывается в этот файл; `--output-dir` дополнительно пишет `<stem>.txt`.
- При `--no-stdout` нужно также передать `--output-file` или `--output-dir`.
- Несколько входов: см. «Пакетные вызовы» выше.

Код выхода:

- `0` — транскрипция завершилась успешно, включая случай пустого или пробельного результата после `str.strip()`.
- `1` — ошибка выполнения без валидного успешного результата; через супервизор `stderr` несёт текст ошибки воркера.
- `2` — неверные аргументы (например `--work-dir` без `--no-daemon`, неизвестный `--priority`).
- `10` — дедлайн вызова (`--client-timeout`) истёк раньше, чем был готов транскрипт; `stderr` начинается с `ASKVLM_CLIENT_TIMEOUT:`.
- `11` — супервизор недоступен, или запрос никогда не уложится в заданные лимиты, или его модель не загрузилась; `stderr` начинается с `ASKVLM_DAEMON_UNAVAILABLE:`.
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
- `--priority normal`

Через супервизор действует политика устройств из раздела «STT-супервизор и пул воркеров», а модели загружаются в строгом режиме устройства: внутри воркера CUDA→CPU fallback не происходит (нехватка памяти GPU проваливает попытку; супервизор повторяет задание `auto` на CPU, а задание `cuda` — на GPU). Правила ниже описывают внутрипроцессный путь модели, которым пользуются legacy `--no-daemon` и остальные команды AskVLM.

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
5. Нехватка памяти GPU на очередном 20-секундном окне переносит модель на CPU, освобождает кэш CUDA и повторяет это же окно на CPU; последующие окна остаются на CPU до перезапуска процесса.
6. Выгрузка модели (`close`/`unload`) после работы на CUDA освобождает кэш CUDA (`torch.cuda.empty_cache`).

На CUDA энкодер работает под fp16 autocast, поэтому текст может незначительно отличаться от CPU-прогона (на 5-минутном тестовом файле — 2 слова из 453). Резидентная модель на CUDA обрабатывает 5-минутный файл примерно в 3.5 раза быстрее, чем на CPU (замер на RTX 3070 с прежним резидентным демоном: около 2.9 с против 10 с на задание с учётом старта клиента); в одноразовом `--no-daemon` на коротких файлах CUDA медленнее CPU из-за инициализации CUDA и child-процесса. Пропускная способность режимов пула измерена в `doc/STT_CONCURRENCY_BENCHMARK.md`.

Параметры Whisper (`--compute-type`, beam/VAD) в GigaAM не передаются.

Так перегруженный GPU не блокирует локальную интеграцию, пока в системе ещё достаточно RAM.

Гарантия fallback относится к стандартному одноразовому пути Whisper и к `gigaam-e2e-ctc` в legacy-режиме `--no-daemon`. Опции вроде диаризации могут потребовать дополнительной ёмкости GPU. `gigaam-ctc` не участвует в CUDA→CPU recovery. Собственного CPU-recovery у клиента нет.

## Надёжность на Windows (subprocess isolation)

Для `external-transcribe --no-daemon` на Windows при `--device != cpu` (Whisper и `gigaam-e2e-ctc`; `gigaam-ctc` всегда работает в процессе на CPU) AskVLM выполняет транскрипцию в child-процессе (через супервизор изоляцию дают процессы-воркеры в Job Object):

1. Parent запускает тот же CLI entrypoint с внутренними скрытыми флагами child-mode.
2. Child пишет служебный JSON-результат до `pipeline.close(...)`.
3. Parent считает этот JSON источником истины:
   - если JSON валиден и содержит `status=ok`, это успех даже при crash-like коде завершения child после записи результата;
   - если валидного успешного JSON нет и код завершения похож на native crash, parent делает ровно один retry на CPU;
   - если валидного успешного JSON нет, ошибка не crash-like и stderr не содержит маркер внутренней IPC-ошибки — parent отдаёт ошибку наружу без CPU retry;
   - если валидного успешного JSON нет и stderr содержит маркер внутренней IPC-ошибки (сбой инициализации child-процесса), parent выполняет один retry на CPU как защитная мера.

## JIT-загрузка и выгрузка

В legacy-режиме `--no-daemon` команда рассчитана на одноразовый запуск (через супервизор воркеры держат модели резидентно до истечения TTL простоя):

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
python cli.py external-transcribe "C:\media\meeting.mp3" --no-daemon --diarization
```

Включить LLM-очистку текста:

```powershell
python cli.py external-transcribe "C:\media\draft.wav" --no-daemon --dialog-blocks
```

Эти опции по умолчанию выключены: они увеличивают стоимость старта и могут подгружать дополнительные ML-бэкенды. Они работают только в legacy-режиме `--no-daemon`; через супервизор команда завершается кодом `2`.

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

Через супервизор дедлайн равен началу вызова + `client_timeout` (по умолчанию 300 с) и включает ожидание в очереди, загрузку модели и транскрипцию. По истечении клиент ставит cancel-маркер; воркер прерывает работу на границе 20-секундного окна (воркер, не остановившийся за срок отмены, убивается).

Вызывающим нужно масштабировать `--client-timeout` с длительностью медиа. Для файлов длиннее нескольких минут значение по умолчанию может оказаться недостаточным: часы аудио на CPU GigaAM не уложатся в 300 с. После таймаута отмена срабатывает кооперативно между окнами, а не мгновенно внутри `model.transcribe`.

## Примечания

- AskVLM хранит кэши моделей в каталоге `.cache/` проекта.
- `external-transcribe` принимает один файл на вызов или пачку файлов через супервизор (`--no-daemon` — один файл).
- `external-extract-frames` рассчитан на один видеофайл на один запуск процесса; кадры остаются в `--output-dir` после завершения.
- Для пакетного экспорта в `txt`, `srt`, `vtt` или `json` используйте `python cli.py transcribe ...`.
