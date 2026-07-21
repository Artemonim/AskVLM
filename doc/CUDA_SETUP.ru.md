# Руководство по настройке CUDA для AskVLM

## Проблема: PyTorch только с CPU

При запуске приложения можно увидеть:

```
CUDA is required for ML processing, but no compatible GPU is available.
```

Это значит, что установлен PyTorch без CUDA-колёс.

Extra `[ml]` (Whisper + GigaAM CTC) рассчитан на **torch/torchaudio 2.10**. Голый `pip install -e ".[ml]"` часто ставит с PyPI **CPU** `2.10.0`.

**По умолчанию:** `run.ps1` / `build.ps1` на каждом запуске чинят стек до CUDA 2.10 (отказ: `-SkipEnsureCUDA`). Отдельный флаг `-EnsureCUDA` больше не нужен.

## Причина

У PyTorch разные бинарные сборки:
- **CPU-only** — работает везде, без GPU
- **CUDA-enabled** — нужен NVIDIA GPU, даёт ускорение

По умолчанию `pip install torch` часто тянет CPU-версию.

## Решение: переустановка PyTorch с CUDA

### Быстрый фикс (автоматически)

```powershell
.\.venv\Scripts\Activate.ps1
.\run.ps1 -SkipLaunch -Fast
```

Или тот же путь через `build.ps1`. Отказ от авточинки: `-SkipEnsureCUDA` / `-SkipEnsureML`.

Скрипт:
1. Проверяет, что torch — **2.10.*** **и** CUDA доступна
2. При необходимости ставит CUDA-колёса PyTorch 2.10 (cu128 → cu126)
3. Верифицирует установку
4. Показывает успех/ошибку

Устаревшие `-EnsureCUDA` / `-EnsureML` всё ещё принимаются, но избыточны (поведение уже включено по умолчанию).

### Ручная установка

Сверьте драйвер:

```powershell
nvidia-smi
```

Смотрите строку "CUDA Version" (это capability драйвера; у PyTorch свой runtime).

#### Вариант 1: CUDA 12.8 (рекомендуется)

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

#### Вариант 2: CUDA 12.6

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.10.0+cu126 torchvision==0.25.0+cu126 torchaudio==2.10.0+cu126 `
  --index-url https://download.pytorch.org/whl/cu126 `
  --extra-index-url https://pypi.org/simple
```

**Примечание:** колёса 12.6/12.8 работают с актуальными драйверами CUDA 12.x. У PyTorch 2.10 в этой линейке нет cu124/cu121 — не смешивайте старый `2.5.1+cu124` со стеком AskVLM `[ml]` (GigaAM требует 2.10).

## Проверка

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Ожидаемый пример:

```
torch: 2.10.0+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX XXXX
```

## Troubleshooting

### PyTorch ставится как CPU, несмотря на CUDA index

**Симптом:** после установки `torch.__version__` показывает `2.10.0` / `2.10.0+cpu` вместо `2.10.0+cu128`.

**Причина:** pip не нашёл CUDA-wheel или упал по сети и откатил на CPU с PyPI. То же после `pip install -e ".[ml]"` без `-EnsureCUDA`.

**Решение:** явный суффикс CUDA в версии пакета:

```powershell
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --no-cache-dir `
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

`build.ps1 -EnsureCUDA` делает то же с откатом cu128 → cu126 и переустанавливает, если нет CUDA **или** torch не `2.10.*`.

### Сетевые проблемы с download.pytorch.org

1. Смените DNS (`1.1.1.1` / `8.8.8.8`), `ipconfig /flushdns`
2. Скачайте `.whl` вручную с https://download.pytorch.org/whl/cu128/torch/ (cp311, `+cu128`) и поставьте через `pip install --no-deps`
3. Проверьте: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### «nvidia-smi not found»

Установите драйвер NVIDIA и перезагрузитесь: https://www.nvidia.com/Download/index.aspx

### CUDA есть, но ML всё равно падает

Проверьте VRAM:

```powershell
python -c "import torch; print('VRAM available:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

Ориентиры: Whisper 1–4 ГБ, диаризация 2–3 ГБ, LLM 2–3 ГБ; GigaAM CTC — ~2.5 ГБ **RAM** (CPU path, без VRAM).

## Системные требования

| Требование | Минимум | Рекомендуется | RTX 30/40 |
|---|---|---|---|
| GPU | GTX 960 | RTX 2070+ | RTX 30/40 |
| VRAM | 6 ГБ | 8–12 ГБ | 8+ ГБ |
| CUDA (драйвер) | 12.x | 12.6+ | 12.8+ |
| PyTorch | 2.10.+cu126 | 2.10.+cu128 | 2.10.+cu128 |

## Ссылки

- https://pytorch.org/get-started/locally/
- https://pytorch.org/get-started/previous-versions/
- https://www.nvidia.com/Download/index.aspx
