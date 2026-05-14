# Настройка CUDA для AskVLM

## Проблема: PyTorch только для CPU

При запуске приложения может появиться следующая ошибка:

```
CUDA is required for ML processing, but no compatible GPU is available.
```

Это происходит, когда PyTorch установлен с поддержкой только CPU вместо CUDA-версии.

## Причина

PyTorch распространяется в нескольких вариантах:
- **Только CPU**: работает на любой системе, но не использует ускорение GPU
- **С поддержкой CUDA**: требует видеокарту NVIDIA, обеспечивает значительное ускорение

По умолчанию команда `pip install torch` часто устанавливает CPU-версию, особенно если окружение системы настроено неправильно.

## Решение: переустановить PyTorch с поддержкой CUDA

### Быстрое исправление (автоматическое)

Используйте скрипт сборки с флагом `-EnsureCUDA`:

```powershell
.\.venv\Scripts\Activate.ps1
.\build.ps1 -EnsureCUDA
```

Скрипт выполнит следующее:
1. Определит, доступна ли CUDA
2. Попытается установить CUDA-версию PyTorch из нескольких репозиториев (cu124, cu121, cu118)
3. Проверит установку
4. Выведет статус успеха или ошибки

### Установка вручную

Выберите версию CUDA, соответствующую вашей системе. Узнать версию CUDA:

```powershell
nvidia-smi
```

Найдите строку «CUDA Capability Major/Minor version» или «CUDA Version» в выводе.

#### Вариант 1: CUDA 12.8 (последняя, рекомендуется для RTX 30/40 серии)

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

#### Вариант 2: CUDA 12.4

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 `
  --index-url https://download.pytorch.org/whl/cu124 `
  --extra-index-url https://pypi.org/simple
```

#### Вариант 3: CUDA 12.1

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 `
  --index-url https://download.pytorch.org/whl/cu121 `
  --extra-index-url https://pypi.org/simple
```

#### Вариант 4: CUDA 11.8 (старые системы)

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 `
  --index-url https://download.pytorch.org/whl/cu118 `
  --extra-index-url https://pypi.org/simple
```

**Примечание:** пакеты CUDA 12.4+ обратно совместимы с драйверами CUDA 12.8. Если у вас установлена CUDA 12.8, пакеты cu124 будут работать нормально, но cu128 обеспечит наилучшую совместимость.

## Проверка

После установки убедитесь, что CUDA настроена корректно:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Ожидаемый вывод:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX XXXX
```

## Устранение неполадок

### Ошибка: PyTorch устанавливается в CPU-версии даже при указании CUDA-индекса

**Симптом:**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Результат: `torch.__version__` показывает `2.9.0+cpu` вместо `2.9.0+cu124`.

**Причина:**
Когда pip не может найти точный CUDA-пакет или сталкивается с сетевыми проблемами (сбои DNS, ошибки SSL), он переключается на CPU-версию из стандартного PyPI.

**Решение:**
Явно укажите суффикс версии CUDA в названии пакета:

```powershell
# * Сначала очистите кэшированные пакеты
pip uninstall -y torch torchvision torchaudio
pip cache purge

# * Установите с явным суффиксом версии CUDA
pip install --no-cache-dir `
  torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

**Почему это работает:**
- Явная версия с суффиксом `+cu128` заставляет pip искать только CUDA-пакеты
- `--no-cache-dir` гарантирует загрузку свежих пакетов без использования потенциально некорректных кэшированных
- `--index-url` задаёт основной индекс как репозиторий PyTorch CUDA
- `--extra-index-url` позволяет загружать зависимости из PyPI

**Автоматическое исправление:**
Скрипт `build.ps1 -EnsureCUDA` теперь автоматически использует этот метод с откатом на несколько версий CUDA (cu128 → cu124 → cu121 → cu118).

### Сетевые проблемы при загрузке PyTorch

**Симптомы:**
- Ошибки `getaddrinfo failed` при загрузке с `download.pytorch.org`
- Ошибки `SSLEOFError` или сбои TLS handshake
- Загрузка работает в браузере, но падает в pip

**Возможные причины:**
1. Проблемы с разрешением DNS в сетевом стеке Python
2. CloudFront отдаёт разные адреса IPv4/IPv6
3. Изменения конфигурации VPN или сети
4. Брандмауэр или антивирус блокирует соединения pip

**Решения:**

1. **Изменить DNS-серверы** (в настройках сети Windows):
   - Укажите DNS `1.1.1.1` (Cloudflare) или `8.8.8.8` (Google)
   - После изменения выполните `ipconfig /flushdns`

2. **Загрузить пакеты вручную** (при сохраняющихся сетевых проблемах):
   ```powershell
   # Скачайте .whl файлы в браузере по адресам:
   # https://download.pytorch.org/whl/cu128/torch/
   # https://download.pytorch.org/whl/cu128/torchvision/
   # https://download.pytorch.org/whl/cu128/torchaudio/
   
   # Ищите файлы cp311-cp311-win_amd64.whl с суффиксом +cu128
   # Пример: torch-2.9.0+cu128-cp311-cp311-win_amd64.whl
   
   # Сохраните в папку wheels/, затем установите:
   pip install --no-deps .\wheels\torch-2.9.0+cu128-cp311-cp311-win_amd64.whl
   pip install --no-deps .\wheels\torchvision-0.24.0+cu128-cp311-cp311-win_amd64.whl
   pip install --no-deps .\wheels\torchaudio-2.9.0+cu128-cp311-cp311-win_amd64.whl
   ```

3. **Проверить установку** после любого способа:
   ```powershell
   python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
   ```

### Ошибка: «nvidia-smi not found»

Это означает, что драйверы NVIDIA не установлены или не добавлены в PATH.

**Решение:**
1. Скачайте последний драйвер NVIDIA: https://www.nvidia.com/Download/driverDetails.aspx
2. Установите драйвер
3. Перезагрузите компьютер
4. Снова выполните `nvidia-smi`

### Ошибка: «CUDA is available but ML still fails»

Возможно, недостаточно видеопамяти (VRAM). Проверьте доступную VRAM:

```powershell
python -c "import torch; print('VRAM available:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

Требования приложения:
- Модель Whisper: 1–4 ГБ в зависимости от размера модели
- Диаризация: 2–3 ГБ
- LLM-форматтер: 2–3 ГБ для модели 7B

При нехватке VRAM используйте модели меньшего размера или уменьшите batch size.

### Ошибка: «Mixed CUDA versions»

Если у вас установлено несколько версий CUDA или тулкитов, PyTorch может использовать неверную.

**Решение:**
```powershell
pip uninstall torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```

## Информация о системе

Проверить GPU и объём VRAM:

```powershell
python -c "import torch; props = torch.cuda.get_device_properties(0); print(f'GPU: {props.name}'); print(f'VRAM: {props.total_memory / 1e9:.1f} GB')"
```

## Дополнительно: системные требования

| Требование | Минимум | Рекомендуется | RTX 30/40 серия |
|---|---|---|---|
| GPU | GTX 960 (Maxwell) | RTX 2070+ | RTX 30/40 серия |
| VRAM | 6 ГБ | 8–12 ГБ | 8+ ГБ |
| Версия CUDA | 11.8 | 12.1–12.4 | 12.4+ |
| Версия драйвера | 450+ | Последняя | Проверьте через nvidia-smi |

## Ссылки

- Официальная установка PyTorch: https://pytorch.org/get-started/locally/
- Матрица совместимости PyTorch и CUDA: https://pytorch.org/get-started/previous-versions/
- Загрузка драйверов NVIDIA: https://www.nvidia.com/Download/index.aspx
