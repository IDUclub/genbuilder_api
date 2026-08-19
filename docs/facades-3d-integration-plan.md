# Интеграция генерации 3D-фасадов (CTLab-ITMO/Facades-3D) — план работ

Документ для передачи задачи другому агенту/разработчику. Содержит проверенные
факты о внешнем сервисе, принятые решения, сделанную часть и остаток работ.

- **Внешний сервис:** [CTLab-ITMO/Facades-3D](https://github.com/CTLab-ITMO/Facades-3D), см. его `API.md`
- **Ветка genbuilder на момент составления:** `feat/mcp_tools`
- **Дата:** 2026-08-17
- **Статус:** сделан этап 0 (экспортёр mass model + CLI для замеров) и часть
  этапа 2 в GenBuilder (HTTP-клиент `facade-jobs`, три REST-зеркала и чат-SSE).
  Инстанс Facades-3D и сервис `facade-jobs` ещё не подняты, ни одной реальной
  генерации не проведено.

---

## 1. Цель

Дать фронту генбилдера (чат) возможность получить не только GeoJSON со зданиями,
но и 3D-сцену квартала с генеративными фасадами (`.glb`). Facades-3D принимает
OBJ-массмодель и отдаёт GLB.

## 2. Принятые решения

| Вопрос | Решение |
|---|---|
| Потребитель | фронт генбилдера (чат) |
| Гранулярность | весь квартал одним запросом |
| Допустимое время | минуты |
| Инстанс Facades-3D | свой, за нашим прокси |
| Прогресс/отмена в UI | не нужны → REST-транспорт к Facades-3D, WebSocket не используем |
| Хранение результатов | MinIO (уже есть) |
| Job-слой | **отдельный сервис** `facade-jobs`, не в genbuilder_api |
| Высота этажа | 3 м (`floors_count × 3`) |
| Стиль | свой промпт на зону → **отдельный запрос к Facades-3D на каждую зону**, склейка GLB на нашей стороне |
| Существующие здания | включать в массмодель вместе со сгенерированными |
| Какие ручки дублируем в 3D | все четыре: `/generate/chat/stream`, `/generate/by_scenario`, `/generate/by_blocks`, `/generate/by_territory` |

## 3. Архитектура

```
  фронт (чат)
     |
     | 1. POST /generate/3d/...  (сигнатура как у обычной ручки)
     v
  genbuilder_api                      <- stateless, БД не заводим
     |  обычная генерация -> BuildingFeatureCollection (+ существующие здания)
     |
     | 2. POST /jobs {buildings: FC, style_by_zone, params}
     v
  facade-jobs                         <- новый сервис: job store + очередь + MinIO
     |  GeoJSON -> OBJ (по одному файлу на зону, общий локальный фрейм)
     |  N x POST /generate  (N = число зон)
     v
  Facades-3D (свой инстанс, за прокси, без авторизации)
     |  <- N x .glb
     |  склейка в один .glb -> MinIO
     v
  фронт: GET /jobs/{id} -> {status, result_url}
```

Почему job-слой снаружи: генбилдер сейчас без БД (нет ни postgres, ни alembic в
`requirements.txt`) и без состояния; тащить в него job store, объектное хранилище
и GPU-квоты — менять природу сервиса. Генбилдер получает только HTTP-клиент к
`facade-jobs` (`httpx` уже в зависимостях) и роутеры-зеркала.

---

## 4. Контракт Facades-3D (проверено по исходникам, не по докам)

### 4.1 Формат OBJ на входе

Источник: `src/gen_facades.py`, `src/gen_wall.py` репозитория Facades-3D.

- **Y-up, метры, земля на `y = 0`.**
- **Стена** — грань, у которой `|normal.y| < 0.01` (`get_wall_sizes`,
  `build_scene`). Нормаль считается как `(p1-p0) × (p2-p0)` (`compute_normal`).
- **Стены только квадами:** `split_long_vertical_face` при `len(face) != 4`
  возвращает грань как есть, без разбиения по `max_wall_aspect_ratio`.
  Треугольная сетка (например из `trimesh.creation.extrude_polygon`) не подходит.
- **Обход вершин стены — против часовой при взгляде снаружи.** `place_wall_mesh`
  берёт нормаль грани и разворачивает сгенерированный фасад через
  `angle = arctan2(-n_x, -n_z)`. При обратном обходе фасады смотрят внутрь дома.
- **Порядок вершин квада:** `низ_i, низ_j, верх_j, верх_i`. `get_wall_size` берёт
  ширину как `|v1-v0|`, высоту как `|v2-v1|` и меняет их местами, если
  `|v1.y - v0.y| >= 0.01`.
- **Крыша** — горизонтальная грань, может быть n-угольником; проходит мимо
  генерации стен и просто триангулируется (`create_plane_from_n_points`,
  работает по контуру без дырок).
- **Группы зданий:** серверный путь вызывает `parse_obj_groups(gen_groups=False)`,
  то есть берёт имена из строк `o <name>`; эти же имена становятся именами узлов
  итоговой сцены GLB (`scene.add_geometry(..., node_name=name)`). Это готовый
  канал для связи «узел GLB → здание генбилдера».
- Парсер строк `v` жёсткий: `_, x, y, z = line.split()` — ровно 4 токена, без
  цветов вершин. Индексы граней только положительные, 1-based.

### 4.2 REST API

```
GET  /health                -> {"status": "ok"}   (только живость приложения)
POST /generate              multipart/form-data -> тело ответа = GLB
```

Обязательные поля: `input_model` (файл OBJ), `pixels_per_meter`, `cluster_count`.
Опционально: `style_reference` (картинка), `prompt`, `negative_prompt`,
`diffusion_steps`, `guidance_scale`, `cross_attention_scale`,
`controlnet_conditioning_scale`, `style_ref_scale`, `max_wall_aspect_ratio`,
`slat_steps`, `slat_cfg_strength`, `border_size`, `trellis_mode`,
`mesh_simplify`, `texture_size`, `texture_brightness_factor`,
`texture_postprocess_shrink_px`, `depth_scale_reference`, `seed`,
`output_filename`, `temp_root`.

Коды: `200` GLB, `400` некорректный вход, `422` валидация FastAPI, `500`
внутренняя ошибка, `503` отмена по сигналу сервера.

### 4.3 Ограничения, которые придётся закрывать нам

- **Один `prompt` и один `style_reference` на запрос.** Стены кластеризуются по
  размерам, а не по зданиям и не по зонам — генератор не знает, где жильё, где
  промка. Отсюда решение «запрос на зону».
- `cluster_count` = число уникальных генераций стены (Stable Diffusion + TRELLIS).
  Это главный множитель времени. От числа зданий время почти не зависит — 200
  домов переиспользуют те же кластеры.
- Одна генерация на процесс, очередь в памяти, без job id, без персистентности,
  без лимита длины очереди, без авторизации.
- Отмена — только разрывом соединения (REST) или `{"type":"cancel"}` (WS), и
  только кооперативно: текущий вызов SD/TRELLIS не прерывается.
- Весь входной файл и весь готовый GLB держатся в памяти сервера целиком.
- `temp_root` — путь на файловой системе сервера, приходящий от клиента.
  **В публичный контракт не пробрасывать.**
- На несколько GPU — по процессу на GPU (`CUDA_VISIBLE_DEVICES`), балансировка
  снаружи. `uvicorn --workers N` не годится.
- Развёртывание нетривиально: git-сабмодули с CUDA-расширениями
  (`diffoctreerast`, `mip-splatting`, `nvdiffrast`), собираются из исходников,
  см. `install/`.

---

## 5. Что уже сделано

### `app/logic/mass_model.py`

GeoJSON (выход генбилдера) → OBJ по контракту выше.

- `build_local_frame(fc) -> LocalFrame` — подбирает UTM-зону
  (`GeoSeries.estimate_utm_crs`) и локальный origin в центре bbox. **Один фрейм
  на всю территорию:** все зоны обязаны экспортироваться в нём, иначе GLB разных
  зон не сойдутся при склейке.
- `group_features_by_zone(fc) -> {zone: fc}` — разрез по `properties.zone`,
  отсутствующая зона → `unknown`.
- `buildings_to_obj(fc, frame, params) -> (obj_text, MassModelStats)`.
  - оси: `x = east`, `y = высота`, `z = -north` (правая тройка, Y-up);
  - высота = `floors_count × floor_height_m` (по умолчанию 3.0); нет/0 этажей →
    фолбэк 1 этаж, счётчик в `stats.fallback_height_features`;
  - контур приводится к CCW через `shapely.orient(sign=1.0)`, дырки становятся CW
    — стены двора получают нормали в сторону двора;
  - сегменты короче `min_segment_length_m` (0.2 м) выбрасываются;
  - стены — квады, крыша — n-угольник по внешнему контуру;
  - **вершины не шарятся между зданиями** — намеренно: в CLI-пути Facades-3D
    (`gen_groups=True`) группы определяются по связным компонентам, и дома,
    делящие вершину, слиплись бы в одно «здание»;
  - имена групп: `gb_<feature id>` для сгенерированных, `po_<physical_object_id>`
    для существующих объектов.
- `MassModelStats`: `buildings`, `walls`, `roofs`, `unique_wall_sizes`,
  `skipped_features`, `fallback_height_features`, `roofs_covering_holes`.
  `unique_wall_sizes` (размеры, округлённые до 0.5 м) — верхняя граница
  осмысленного `cluster_count`.

Известные ограничения модуля:
- крыша над внутренним двором закрывает двор (в OBJ у грани не бывает дырок);
  случай считается в `roofs_covering_holes`;
- рельефа нет, все дома стоят на `y = 0`.

### `scripts/export_mass_model.py`

CLI для замеров: режет коллекцию по зонам в общем фрейме, пишет `<zone>.obj`,
печатает таблицу статистики, по флагу `--send-to` гоняет `POST /generate` с
замером времени и сохраняет `.glb`. Промпты по зонам — словарь `ZONE_PROMPTS`
(временно здесь, потом переезжает в `facade-jobs`).

```bash
python scripts/export_mass_model.py --input buildings.geojson --outdir out --send-to http://facades-host:8000 --cluster-count 12
```

### `tests/unit/test_mass_model.py`

9 тестов: квады у стен, нормали стен наружу, нормаль крыши вверх, ширина/высота
стен, MultiPolygon в одной группе, фолбэк без `floors_count`, имена групп,
совпадение координат при разрезе по зонам, отказ на пустой коллекции.
`pytest tests/unit` — 28 passed.

---

## 6. Этап 0 (следующий шаг): замеры на реальном квартале

Блокер: нужен поднятый инстанс Facades-3D.

Замерить на 1–2 реальных кварталах:

1. время генерации при `cluster_count` = 4 / 8 / 16 / 32 (по одной зоне);
2. размер итогового GLB и время его передачи;
3. пиковую VRAM и RAM на инстансе;
4. визуальное качество: достаточно ли 8–12 кластеров, заметны ли повторы.

Решения, которые зависят от замеров: реальные потолки лимитов (раздел 8),
осмысленность разреза по зонам (время × число зон), нужен ли разрез квартала на
части, какой ставить таймаут job.

Пока замеров нет — **любая оценка времени в этом документе умозрительная**.
Ориентир, не подтверждённый практикой: 15–60 с на кластер, то есть
`cluster_count=12` ≈ 3–12 минут на зону.

## 7. Этап 1: сервис `facade-jobs`

Отдельный репозиторий/сервис. Стек по правилам проекта: FastAPI + PostgreSQL +
Alembic, или Redis, если решите, что job-состояние переживать рестарт не обязано
(тогда обосновать). Результаты — MinIO.

### 7.1 API

```
POST   /jobs                -> 202 {job_id, status}
GET    /jobs/{job_id}       -> {status, zones: [...], result_url, error, timings}
DELETE /jobs/{job_id}       -> отмена (best-effort: разрыв соединения к Facades-3D)
GET    /health
```

Тело `POST /jobs`:

```json
{
  "buildings": { "type": "FeatureCollection", "features": [] },
  "floor_height_m": 3.0,
  "style_by_zone": {
    "residential": {"prompt": "...", "style_reference_key": null}
  },
  "params": {"cluster_count": 12, "pixels_per_meter": 32, "seed": null},
  "requested_by": "<user_id>"
}
```

Статусы job: `queued`, `running`, `succeeded`, `failed`, `cancelled`.
Пер-зонные подстатусы — чтобы фронт видел «3 из 4 зон готово».

### 7.2 Пайплайн воркера

1. `build_local_frame` по всей коллекции (один фрейм на job).
2. `group_features_by_zone`, для каждой зоны `buildings_to_obj`.
3. Последовательно (GPU один) `POST /generate` на каждую зону со своим промптом.
4. Склейка N GLB в один: `trimesh.load(..., force='scene')` + перенос узлов в
   общую сцену. Имена узлов уникальны за счёт id зданий, префиксовать не нужно;
   при коллизии — падать, а не молча переименовывать.
5. Загрузка результата в MinIO, `result_url` — presigned ссылка с TTL.
6. Ретраи: только на сетевых ошибках и `503`; `400`/`422` — ошибка входа, ретрай
   бессмысленен.

Код `app/logic/mass_model.py` переносится в этот сервис (или выносится в общую
библиотеку). В генбилдере он после переноса не нужен — но до этапа 2 пусть
остаётся, на нём держится CLI для замеров.

### 7.3 Обязательно предусмотреть

- один активный job на пользователя;
- таймаут job и различение «висит» от «работает»;
- логирование `zone`, `cluster_count`, времени и размера GLB — база для тюнинга;
- `temp_root` в запросы к Facades-3D не передавать.

## 8. Этап 2: ручки-зеркала в genbuilder_api

**Реализовано 2026-08-17:** клиент `app/infrastructure/facade_jobs_client.py`,
orchestration-функции, три REST-ручки с ответом `202`, чат-ручка с событием
`facade_job`, конфигурация `FACADE_JOBS_API` / `FACADE_JOBS_PUBLIC_API` /
`FACADE_JOBS_TIMEOUT_SECONDS` и unit-тесты клиента. Для фактического запуска
остаётся развернуть этап 1 и прописать URL сервиса.

Новые роутеры, существующие `/generate/*` **не меняем**.

| Новая ручка | Дублирует | Возврат |
|---|---|---|
| `POST /generate/3d/by_scenario` | `/generate/by_scenario` | `202 {job_id, status_url}` |
| `POST /generate/3d/by_blocks` | `/generate/by_blocks` | `202 {job_id, status_url}` |
| `POST /generate/3d/by_territory` | `/generate/by_territory` | `202 {job_id, status_url}` |
| `POST /generate/chat/stream/3d` | `/generate/chat/stream` | тот же SSE + событие `facade_job` с `job_id` в конце |

Требования:

- параметры и авторизация — как у оригинальных ручек (`app/utils/auth`);
- логика в `app/logic/generation_orchestration`, в роутерах только HTTP;
- клиент к `facade-jobs` — в `app/infrastructure/` рядом с
  `chat_storage_client.py` / `vllm_chat_client.py`, на `httpx`;
- конфиг через `iduconfig.Config` (как `LLM_API`): `FACADE_JOBS_API` и таймауты.
  Не настроено → `503` с внятным текстом, как это уже сделано для чата;
- **синхронно ждать GLB в хендлере нельзя** — минуты работы плюс общая очередь
  делают время неограниченным; отдаём `job_id`, фронт опрашивает `facade-jobs`.

Стоит обсудить: дублировать ли 3D-ручки в MCP-тулзы (`app/mcp_server/tools/`) —
там уже есть обёртки над `/generate/*`.

## 9. Лимиты (предложение, уточнить после замеров)

Facades-3D не ограничивает ничего. Зашить в `facade-jobs`:

| Параметр | Дефолт | Потолок | Почему |
|---|---:|---:|---|
| `cluster_count` (на зону) | 12 | 32 | прямой множитель времени |
| `pixels_per_meter` | 32 | 64 | их же значение в примерах; выше — VRAM |
| зданий в запросе | — | 300 | размер GLB и время сборки сцены |
| размер OBJ | — | 50 МБ | входной файл читается в память целиком |
| таймаут job | 45 мин | — | иначе висяк не отличить от работы |
| активных job на пользователя | 1 | — | GPU один, очередь общая |

## 10. Риски

1. **Время.** Не измерено. Если `cluster_count=12` даёт больше ~10 минут на
   зону, «минуты» для квартала с 3 зонами не выполняются — придётся снижать
   кластеры, отказываться от разреза по зонам или ставить второй GPU.
2. **Размер GLB.** Стены инстанцируются как отдельные меши и конкатенируются;
   для квартала на 200 зданий файл может выйти в сотни МБ. Влияет и на MinIO, и
   на способность фронта это открыть.
3. **Развёртывание Facades-3D** — CUDA-расширения из исходников. Отдельная
   задача, блокирует всё остальное.
4. **Безопасность.** Сервис без авторизации, лимитов и с клиентским `temp_root`.
   Только приватная сеть, только через `facade-jobs`.
5. **Потеря работы при рестарте** инстанса фасадов: активные и очередные job
   пропадают. `facade-jobs` должен помечать такие job как `failed` по таймауту.
6. **Крыши над дворами** и отсутствие рельефа — известные упрощения массмодели.

## 11. Проверки

```bash
python -m pytest tests/unit -q
```

`ruff`, `black`, `mypy` в проекте не установлены (в `requirements-dev.txt` только
`pytest`), поэтому линт и типы не проверялись — при добавлении инструментов
прогнать по `app/logic/mass_model.py` и `scripts/export_mass_model.py`.

## 12. Журнал проверок инстанса Facades-3D

### 2026-08-18 — `http://a6k4.dgx:8030`, генерация не работает

| Проверка | Результат |
|---|---|
| `GET /health` | `200 {"status":"ok"}` |
| `POST /generate`, пустой файл | `400 The uploaded OBJ file is empty` |
| `POST /generate`, `cluster_count=0` | `400 cluster_count must be positive` |
| `POST /generate`, **их собственный** `example_data/cube.obj`, `pixels_per_meter=16`, `cluster_count=2` | `500 Generation failed: argument of type 'NoneType' is not iterable` за ~1 с |
| то же с `pixels_per_meter=32`, `cluster_count=1` | тот же `500` |
| наш массмодель (3 здания, 2 зоны) | тот же `500` |

Выводы:

- транспорт, multipart и слой валидации исправны — доходит до самой генерации;
- **наш формат OBJ ни при чём**: эталонный `cube.obj` из их репозитория падает так
  же;
- падение за ~1 с (первый запрос — 5 с) — до какой-либо инференс-работы, похоже на
  ленивую инициализацию моделей, которая падает и дальше отдаёт ошибку сразу;
- `argument of type 'NoneType' is not iterable` — это оператор `in` по `None`. В их
  собственном коде генерации такого выражения нет (проверено grep по `src/`),
  значит исключение поднимается внутри библиотеки — diffusers / transformers /
  TRELLIS.

Гипотезы (не проверены, нужен доступ к логам инстанса):

1. веса не подтянулись — `aux/buildingface.safetensors` лежит в репозитории через
   git-lfs, при клоне без lfs вместо весов окажется текстовый указатель;
2. несовместимая версия `diffusers`/`transformers` с кодом загрузки LoRA и
   IP-Adapter;
3. не скачалась модель TRELLIS (`microsoft/TRELLIS-image-large`).

Что нужно от админа инстанса: полный traceback из логов контейнера. Сервер пишет
его сам — `logger.error("Unhandled error while processing POST /generate\n%s",
error.traceback_text)` в `uvicorn.error`. По traceback причина определяется за
минуту, без него — гадание.

Воспроизведение одной командой:

```bash
curl -X POST http://a6k4.dgx:8030/generate -F "input_model=@example_data/cube.obj" -F "pixels_per_meter=16" -F "cluster_count=2" -o cube.glb
```

### 2026-08-18, продолжение — причина найдена, оба бага в публичном репозитории

Логи контейнера `idu-facades-3d` дали traceback. Оба дефекта воспроизводятся по
исходникам `CTLab-ITMO/Facades-3D@main`, приватного форка искать не нужно.

**Баг 1 — `style_ref_scale=0.0` не отключает IP-Adapter.**
`gen_wall.py:55` грузит адаптер безусловно при импорте модуля, а картинка
передаётся только при `use_style_reference = style_ref is not None and
style_ref_scale > 0.0` (`gen_wall.py:144`, `:165-166`). UNet с загруженным
`encoder_hid_proj` требует `added_cond_kwargs["image_embeds"]` независимо от
масштаба, поэтому запрос без `style_reference` падает в diffusers:
`unet_2d_condition.py`, `process_encoder_hidden_states`,
`TypeError: argument of type 'NoneType' is not iterable`.
Поведение, описанное в их `API.md` §4.5, в этой версии diffusers нереализуемо.

**Баг 2 — `NameError` в очистке CUDA.** `gen_wall.py:76` использует голое имя
`SKIP_MODEL_LOAD`, которое нигде не определено (в строках 36 и 61 флаг читается
как `os.getenv("SKIP_MODEL_LOAD")`). `release_generation_memory()` вызывается в
`finally` после каждого запроса (`server.py:226`), а `server.py:227-235`
превращает исключение из cleanup в ошибку запроса. Клиент получает 500 **даже при
успешной генерации**, и CUDA-память при этом не освобождается.

**Проверка, разделившая баги.** `cube.obj` + `example_data/style1.png`,
`style_ref_scale=0.5`, `pixels_per_meter=16`, `cluster_count=1`:

```
HTTP 500 за 38.7 с
{"detail":"Generation failed: CUDA cleanup failed: name 'SKIP_MODEL_LOAD' is not defined"}
```

То есть пайплайн работает: SD + TRELLIS собрали стену и сцену примерно за 38 с,
результат уничтожен на этапе очистки. Первый замер: ~38 с на один кластер при
`pixels_per_meter=16` (включая постоянные накладные расходы). При `ppm=32` будет
дольше.

**Обход на нашей стороне:** всегда передавать `style_reference` **и**
`style_ref_scale > 0` — иначе их код не отдаст картинку в пайплайн. Это снимает
баг 1, но не баг 2, поэтому до исправления образа успешных ответов не будет.

**Побочный эффект бага 2:** после каждого падения VRAM не освобождается. Серия
неудачных запросов способна довести GPU до OOM; лечится рестартом контейнера.

**Фиксы (по две строки):**

```python
# gen_wall.py, вверху модуля — и использовать в :36, :61, :76
SKIP_MODEL_LOAD = bool(os.getenv("SKIP_MODEL_LOAD"))
```

```python
# gen_wall.py:165 — адаптер загружен всегда, значит картинка нужна всегда
pipeline_kwargs["ip_adapter_image"] = (
    style_ref if use_style_reference else Image.new("RGB", (224, 224), "white")
)
```

Открытый вопрос: чиним патчем в своём образе, заводим PR в CTLab-ITMO или и то и
другое.

### Патч образа (без апстрима)

Решено чинить только в своём контейнере, PR в CTLab-ITMO не заводить. Патч лежит
в репозитории genbuilder: `scripts/patch_facades3d_gen_wall.py` — идемпотентный,
делает `.orig`-бэкап, падает с ошибкой, если не нашёл якорь. Правит те же два
места: определяет `SKIP_MODEL_LOAD` и всегда передаёт `ip_adapter_image`.

```bash
docker exec idu-facades-3d sh -c 'find / -name gen_wall.py -not -path "*/site-packages/*" 2>/dev/null'
docker cp scripts/patch_facades3d_gen_wall.py idu-facades-3d:/tmp/patch.py
docker exec idu-facades-3d python /tmp/patch.py <путь>/gen_wall.py
docker restart idu-facades-3d
```

**Правка живёт в writable-слое контейнера.** `docker restart` её сохраняет;
`docker compose up --force-recreate`, `docker rm` и обновление образа — стирают,
и баг вернётся молча. Чтобы закрепить: `docker commit idu-facades-3d
idu-facades-3d:patched` и перевести compose на этот тег, либо примонтировать
пропатченный `gen_wall.py` бинд-маунтом.

Приёмочный тест — тот же `cube.obj`, но уже **без** `style_reference`:

```bash
curl -X POST http://a6k4.dgx:8030/generate -F "input_model=@example_data/cube.obj" -F "pixels_per_meter=16" -F "cluster_count=1" -o cube.glb -w "%{http_code}\n"
```

Ожидание: `200` и валидный GLB вместо `500`.
