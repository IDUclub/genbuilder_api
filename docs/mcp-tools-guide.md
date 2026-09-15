# GenBuilder MCP — гайд по тулзам

Практическое руководство по MCP-серверу GenBuilder: какие тулзы есть, как к ним
обращаться и по какому адресу сервер живёт. MCP-сервер смонтирован in-process на
том же FastAPI-приложении, что отдаёт REST (`app/main.py`), и вызывает тот же
слой оркестрации (`app.logic.generation_orchestration`) — поведение идентично
классическим `/generate/*` эндпоинтам, см. [frontend-api-guide.md](frontend-api-guide.md).

- **Транспорт:** MCP streamable-HTTP (JSON-RPC 2.0 поверх HTTP/SSE)
- **Путь:** `/mcp`
- **Версия:** GenBuilder API `0.1.1`

> ✅ **Статус на проде: живой.** MCP-сервер реализован на ветке `feat/mcp_tools`
> (`app/mcp_server/`), которая ещё **не смержена** в `main`/`dev`, но уже
> **задеплоена** на `http://10.32.1.46:8200`. Проверено вручную 2026-07-24:
> `POST /mcp/` (`initialize`) → `200 OK` с `serverInfo: "GenBuilder MCP"`,
> `tools/list` отдавал 4 тулзы (всё ниже, кроме добавленной позже `list_functional_zones`) — идентично локальной Docker-сборке этой же
> ветки. Раньше в тот же день на этом хосте `/mcp` отдавал чистый `404` — то есть
> прод обновляется отдельными выкатками, а не сразу при пуше в ветку; если снова
> увидишь `404` без редиректа на `/mcp` и `/mcp/` — деплой мог откатиться или уйти
> на другой хост, стоит перепроверить (см. [4.2](#42-сырой-json-rpc-curl)).
>
> **Важно:** `openapi.json` в принципе никогда не покажет `/mcp` — это смонтированное
> Starlette-под-приложение (`app.mount("/mcp", mcp_app)`), FastAPI не включает такие
> роуты в свою OpenAPI-схему. Отсутствие `/mcp` в `openapi.json` само по себе ничего
> не доказывает; решающий тест — прямой запрос на `/mcp`.

---

## 1. Адрес

| Окружение | URL |
|---|---|
| Прод | `http://10.32.1.46:8200/mcp` |
| Локально | `http://localhost:8000/mcp` |

Монтирование — [app/main.py:89-90](../app/main.py#L89-L90):

```python
# MCP tools (see app/mcp_server) — streamable-HTTP transport at /mcp.
app.mount("/mcp", mcp_app)
```

---

## 2. Авторизация

Как и в REST — **Keycloak Bearer-токен** пользователя (realm `IDU`), берётся из
заголовка `Authorization` входящего HTTP-запроса и форвардится в UrbanDB
([app/mcp_server/auth.py](../app/mcp_server/auth.py)):

```
Authorization: Bearer <keycloak_access_token>
```

- Нужен всем тулзам, кроме `generate_by_territory` (работает только с инлайн-геометрией).
- Токен **короткоживущий** (обычно 300 сек / 5 мин — TTL задаётся Keycloak-realm'ом).
  Если между вызовами есть пауза (например, уточняющий вопрос пользователю) — токен
  может протухнуть, тогда нужен свежий.
- Просроченный/неверный токен → JSON-RPC ошибка `-32002 AUTH_TOKEN_EXPIRED` (см. [раздел 5](#5-ошибки)).
- Сервер **ничего не кэширует**: токен читается из заголовка `Authorization`
  заново на каждом вызове тулзы. Поэтому в длинной цепочке (оркестратор гоняет
  `list_functional_zones` → `estimate_…` → `generate_…`, между шагами — LLM и
  вопросы пользователю) оркестратор должен **обновлять токен перед каждым
  вызовом** (refresh token в Keycloak), а не держать один access token на весь
  план. На `-32002` — обновить токен и повторить тот же вызов.

---

## 3. Тулзы

Определены в [app/mcp_server/tools/generation.py](../app/mcp_server/tools/generation.py).

Рекомендуемый порядок работы агента: `list_functional_zones` →
`estimate_max_residents_by_blocks` (реалистичные targets) → `generate_by_blocks` /
`generate_by_scenario`.

> **Отличия MCP от REST.**
> - MCP **не подставляет дефолтные targets молча**. Нужно либо передать
>   `targets_by_zone`, либо явно указать `use_defaults: true`. Иначе вернётся `-32602`.
> - Каждый результат генерации содержит `summary` (см. [3.6](#36-summary-в-ответе-генерации)).
> - Слой зданий **не возвращается целиком** по умолчанию: он сохраняется в объектное
>   хранилище, а в ответе приходят `result_id` и ссылка `layer`
>   (см. [3.7](#37-хранение-результата-и-воспроизводимость)).

### 3.1. `list_functional_zones`

Список функциональных зон сценария: id, тип и площадь. Нужен, чтобы выбрать
`functional_zone_ids` для `generate_by_blocks` / `estimate_max_residents_by_blocks`.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `scenario_id` | int | ✅ | ID проекта/сценария |
| `year` | int | ✅ | Год данных функциональных зон |
| `source` | string | ✅ | Источник зон, напр. `"OSM"`, `"PZZ"`, `"User"` |
| `functional_zone_types` | list[string] | ⛔ | Оставить только зоны этих типов |

**Auth:** нужен bearer-токен. Read-only.
**Возвращает:**

```json
{
  "scenario_id": 843, "year": 2025, "source": "User",
  "zones": [
    { "functional_zone_id": 6679027, "functional_zone_type": "residential_midrise",
      "generation_zone": "residential", "name": null,
      "geometry_type": "Polygon", "area_m2": 84210.5 }
  ],
  "totals_by_type": { "residential_midrise": { "count": 1, "area_m2": 84210.5 } }
}
```

`generation_zone` — каноническая зона, в которую тип зоны превращается при
генерации (`residential_*` → `residential`, `mixed_use` → `business`). Площадь
считается в локальной UTM-проекции. Пустой `zones` означает, что для этих
year/source зон нет. Какие year/source вообще доступны, gateway не отдаёт —
их нужно знать заранее.

### 3.2. `generate_by_scenario`

Генерация зданий по всей территории сценария UrbanDB.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `scenario_id` | int | ✅ | ID проекта/сценария |
| `year` | int | ✅ | Год данных функциональных зон |
| `source` | string | ✅ | Источник зон, напр. `"OSM"`, `"PZZ"`, `"User"` |
| `functional_zone_types` | list[string] | ✅ | Типы зон для генерации, напр. `["residential", "business", "industrial"]` |
| `targets_by_zone` | object | ✅* | Спрос по зонам (residents / coverage_area / floors_avg / density_scenario / default_floor_group). *Обязателен, если не задан `use_defaults: true` |
| `use_defaults` | bool | ⛔ | `true` — явно генерировать на дефолтных targets сервиса. Использовать только с согласия пользователя |
| `preserve_existing_buildings` | bool | ⛔ | `true` — сохранить существующие здания сценария: они вырезаются из территории и возвращаются с `is_excluded: true`. Если здания не удалось загрузить, генерация **не запускается** (`-32603`) |
| `physical_object_id` | list[int] | ⛔ | Id физ. объектов, исключить из территории |
| `generation_parameters` | object | ⛔ | Низкоуровневые оверрайды генерации (напр. `{"rectangle_finder_step": 5}`). Невалидные значения → `-32602` до запуска генерации |
| `seed` | int | ⛔ | Seed случайной расстановки сервисов. Не задан — выбирается случайно и возвращается в ответе |
| `include_geometry` | bool | ⛔ | `true` — дополнительно вернуть `features` прямо в ответе. По умолчанию `false` |

**Auth:** нужен bearer-токен.
**Возвращает:** результат генерации (см. [3.7](#37-хранение-результата-и-воспроизводимость)):
`summary`, `result_id`, `layer`, `seed`, `applied_parameters`, `generation_id`,
`duration_s`. Сам слой — GeoJSON `FeatureCollection` сгенерированных и исключённых
зданий (в `properties` каждой фичи — `floors_count`, `living_area`, `functional_area`,
`building_area`, `zone`, `service`).

Зданием считается физ. объект с записью `building` или с типом «жилой дом»
(`physical_object_type_id = 4`) и полигональной геометрией. Точечные здания
вырезать из квартала нельзя, поэтому они пропускаются.

### 3.3. `generate_by_territory`

Генерация по присланным полигонам блоков — без привязки к сценарию.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `blocks` | GeoJSON FeatureCollection | ✅ | `Polygon`/`MultiPolygon`-фичи, у каждой заполнен `properties.zone` (напр. `"residential"`) |
| `targets_by_zone` | object | ✅* | Как выше, *или `use_defaults: true` |
| `use_defaults` | bool | ⛔ | Как выше |
| `existing_buildings` | GeoJSON FeatureCollection | ⛔ | Пятна уже стоящих зданий — вырезаются из блоков до генерации и возвращаются с `is_excluded: true`; `properties` необязательны |
| `territory_id` | int | ⛔ | Регион UrbanDB: по его нормативам в жилых блоках расставляются сервисы (школы, детские сады и т. п.). Без него сервисов нет |
| `generation_parameters` | object | ⛔ | Как выше |
| `seed`, `include_geometry` | — | ⛔ | Как выше |

**Auth:** не требуется.
**Возвращает:** результат генерации, как у `generate_by_scenario`; в `summary` нет
`existing_buildings_preserved`.
**Ошибка:** `-32602 Invalid params`, если нет ни `targets_by_zone`, ни `use_defaults: true`, либо геометрия блока отсутствует, не `Polygon`/`MultiPolygon` или нет `zone`.

### 3.4. `generate_by_blocks`

Генерация по выбранным functional zone id внутри сценария — свои targets на
каждую зону, один прогон на зону (или на часть полигона, если зона — MultiPolygon).

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `scenario_id`, `year`, `source`, `functional_zone_types` | — | ✅ | Как в `generate_by_scenario` |
| `zones` | list[object] | ✅ | `[{ functional_zone_id, targets_by_zone, generation_parameters? }, ...]` — по записи на зону |
| `preserve_existing_buildings` | bool | ⛔ | Сохранить существующие здания внутри запрошенных зон (как выше) |
| `physical_object_id` | list[int] | ⛔ | Как выше |
| `seed` | int | ⛔ | Общий seed, подставляется в `generation_parameters` каждой зоны |
| `include_geometry` | bool | ⛔ | Как выше |

**Auth:** нужен bearer-токен.
**Возвращает:** результат генерации, как у `generate_by_scenario`; слой объединяет
здания всех запрошенных зон. Targets в `summary` — сумма targets по всем зонам.
`applied_parameters` — `{ "zones": [{ functional_zone_id, generation_parameters, targets_by_zone }] }`.
**Прогресс:** если запрос пришёл с `progressToken`, после каждой зоны отправляется
`notifications/progress` (`progress`/`total` — число готовых/всех зон).
**Ошибки:** `-32602`, если `functional_zone_id` не существует для этого
сценария/года/источника, либо геометрия зоны не Polygon/MultiPolygon; `-32603`,
если при `preserve_existing_buildings: true` не удалось загрузить здания.

### 3.5. `estimate_max_residents_by_blocks`

Оценка вместимости зон на дефолтных (максимально плотных) targets — без
полного построения застройки.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `scenario_id`, `year`, `source`, `functional_zone_types` | — | ✅ | Как выше |
| `functional_zone_ids` | list[int] | ✅ | Зоны для оценки |
| `preserve_existing_buildings` | bool | ⛔ | `true` — сначала вырезать существующие здания, чтобы оценка показала **дополнительную** вместимость свободной земли |

**Auth:** нужен bearer-токен. Read-only (`annotations.readOnlyHint: true`).
**Возвращает:**

```json
{
  "zones": [
    { "functional_zone_id": 6679027, "functional_zone_type": "residential",
      "zone_area_m2": 84210.5, "max_residents": 2350, "max_living_area": 70500.0,
      "existing_buildings_count": 12, "existing_living_area": 18400.0, "existing_residents": 610 }
  ],
  "totals": { "zone_area_m2": 84210.5, "max_residents": 2350, "max_living_area": 70500.0,
              "existing_buildings_count": 12, "existing_living_area": 18400.0, "existing_residents": 610 },
  "existing_buildings_preserved": false,
  "duration_s": 12.4
}
```

Поля `existing_*` заполняются всегда, независимо от флага. Прогресс по зонам —
как у `generate_by_blocks`. REST-эндпоинт
`/generate/max_residents_by_blocks` по-прежнему отдаёт старый формат
`{ <functional_zone_id>: <residents_count> }`.

### 3.6. `summary` в ответе генерации

```json
"summary": {
  "buildings": 42, "living_area_total": 61234.5, "residents_total": 2040,
  "buildings_by_zone": { "residential": 30, "business": 12 },
  "residents_by_zone": { "residential": 2040, "business": 0 },
  "excluded_buildings": 7, "existing_living_area": 9800.0, "existing_residents": 320,
  "targets": {
    "residential": { "target_residents": 3000, "achieved_residents": 2040, "residents_deficit": 960 },
    "business": { "target_functional_area": 20000.0, "achieved_functional_area": 18500.0, "functional_area_deficit": 1500.0 }
  },
  "targets_source": "request",
  "existing_buildings_preserved": true
}
```

- Поля `buildings`, `*_total` и `*_by_zone` считаются **только по новым** зданиям.
  Исключённые (существующие) здания учитываются отдельно — в `excluded_buildings` и `existing_*`.
- `targets` содержит плановые и фактические значения и дефицит по каждой зоне
  (дефицит не бывает отрицательным).
- `targets_source` — откуда взяты targets: `"request"` или `"service_defaults"`.

### 3.7. Хранение результата и воспроизводимость

Слой зданий крупного сценария весит мегабайты — в контекст LLM-агента его не
тащим. Все три `generate_*` сохраняют полный `FeatureCollection` в объектное
хранилище и возвращают:

```json
{
  "generation_id": "3f9c0a1e6b2d4c7f9e8a1b2c3d4e5f60",
  "result_id": "3f9c0a1e6b2d4c7f9e8a1b2c3d4e5f60",
  "layer": {
    "name": "buildings", "title": "...", "role": "...",
    "url": "https://<host>/files/buildings/3f9c0a1e6b2d4c7f9e8a1b2c3d4e5f60",
    "download_url": null, "filename": "buildings.geojson",
    "mime_type": "application/geo+json", "source_service": "genbuilder"
  },
  "summary": { "...": "см. 3.6" },
  "seed": 1834201775,
  "applied_parameters": { "generation_parameters": { "...": "..." }, "targets_by_zone": { "...": "..." } },
  "duration_s": 48.217
}
```

- `layer.url` — ссылка на слой для карты или другого сервиса (эндпоинт `/files`,
  нужен тот же bearer-токен). Без агента слой можно забрать через
  [`get_generation_result`](#38-get_generation_result).
- `include_geometry: true` добавляет в ответ `type` и `features`.
- Если хранилище не настроено или запись упала, `result_id` и `layer` равны `null`,
  `features` возвращаются в ответе, а в `storage_warning` — причина. Генерация
  при этом не теряется.
- Результаты хранятся ограниченное время — долгоживущий план должен сохранять
  нужное у себя, а не рассчитывать на `result_id` через дни.

**Воспроизводимость.**

- `seed` — seed случайной расстановки сервисов. Повторный вызов с теми же входными
  данными и тем же `seed` даёт ту же раскладку. Остальные шаги генерации
  детерминированы и seed не используют.
- `applied_parameters` — **эффективные** параметры: дефолты сервиса с наложенными
  оверрайдами (включая `seed`) и targets, которые реально ушли в генерацию.
  По ним видно, что именно считалось, даже если targets взяты из `use_defaults`.
  Для `generate_by_blocks` — список по зонам.
- `generation_id` — id прогона для логов и ссылок между шагами; совпадает с
  `result_id`, если результат сохранён.
- `duration_s` — время выполнения тулзы в секундах (есть и у
  `estimate_max_residents_by_blocks`).

**Длинные цепочки.**

- Токен читается из `Authorization` **на каждый вызов** и живёт ~5 минут. Оркестратор
  должен обновлять его между шагами, а на `-32002` — брать свежий и повторять.
- `generate_by_blocks` и `estimate_max_residents_by_blocks` шлют
  `notifications/progress` после каждой зоны, если в запросе есть `progressToken`.
- Отмена запроса (`notifications/cancelled`) останавливает обработку на границе
  следующей зоны: зона, которая уже считается в рабочем потоке, дорабатывает, но
  следующая не запускается, и результат не сохраняется.

### 3.8. `get_generation_result`

Чтение сохранённого результата генерации по `result_id`.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `result_id` | string | ✅ | `result_id` из ответа генерации |
| `include_geometry` | bool | ⛔ | По умолчанию `true`. `false` — только метаданные (`summary`, `seed`, `applied_parameters`, …) без `features` |

**Auth:** нужен bearer-токен. Read-only.
**Возвращает:** `FeatureCollection` с полями `generation_id`, `summary`, `seed`,
`applied_parameters`, `duration_s` на верхнем уровне.
**Ошибки:** `-32602` — некорректный `result_id` или результат не найден (истёк);
`-32603` — хранилище не настроено или недоступно.

> **Совместимость со слоями Provision / PzzCompare** не проверена: схемы входных
> слоёв этих сервисов нам недоступны. `layer` оформлен так же, как остальные слои
> GenBuilder (`/files/<slot>/<id>`), но принимают ли они такой дескриптор и формат
> `properties` зданий — нужно сверить по их схемам.

---

## 4. Как обращаться

### 4.1. Через MCP-клиент (рекомендуется)

Любой клиент со streamable-HTTP transport (Claude Desktop, `fastmcp`, LangChain
MCP adapter и т.п.):

```json
{
  "mcpServers": {
    "genbuilder": {
      "url": "http://10.32.1.46:8200/mcp",
      "headers": { "Authorization": "Bearer <keycloak_access_token>" }
    }
  }
}
```

Python (`fastmcp`):

```python
from fastmcp import Client
import asyncio

async def main():
    async with Client("http://10.32.1.46:8200/mcp", auth="<bearer_token>") as client:
        tools = await client.list_tools()
        result = await client.call_tool(
            "generate_by_scenario",
            {
                "scenario_id": 843,
                "year": 2025,
                "source": "User",
                "functional_zone_types": ["residential", "business", "industrial"],
                "targets_by_zone": {"residents": {"residential": 3000}},
                "preserve_existing_buildings": True,
            },
        )
        generated = result.structured_content
        print(generated["summary"], generated["seed"])

        # полный слой — только когда он действительно нужен
        layer = await client.call_tool(
            "get_generation_result", {"result_id": generated["result_id"]}
        )
        print(len(layer.structured_content["features"]))

asyncio.run(main())
```

### 4.2. Сырой JSON-RPC (curl)

Нужен MCP-handshake перед вызовом тулзы: сначала `initialize` (ответ содержит
`Mcp-Session-Id` — передавай его в заголовке на все последующие вызовы), затем
`tools/call`. Обязательны оба `Accept`-типа.

`POST /mcp` (без слэша) отдаёт `307 → /mcp/` — либо бей сразу в `/mcp/`, либо
добавляй `-L`:

```bash
curl -s -X POST http://10.32.1.46:8200/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0.1"}}}'
# -> забери "mcp-session-id" из заголовков ответа

curl -s -X POST http://10.32.1.46:8200/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <session_id из initialize>" \
  -H "Authorization: Bearer <keycloak_access_token>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "generate_by_scenario",
      "arguments": {
        "scenario_id": 843,
        "year": 2025,
        "source": "User",
        "functional_zone_types": ["residential", "business", "industrial"],
        "use_defaults": true,
        "preserve_existing_buildings": true
      }
    }
  }'
```

Если `404 Not Found` (без редиректа) и на `/mcp`, и на `/mcp/` — MCP-сервер на
этом хосте не смонтирован (см. предупреждение в начале файла), а не проблема
с запросом.

### 4.3. Эквивалент через REST

Тот же результат (и то же поведение при ошибках) даёт классический эндпоинт —
полезно, если нужен обычный REST-клиент без MCP-транспорта, либо на случай,
если `/mcp` на конкретном хосте временно недоступен (подробнее —
[frontend-api-guide.md §5](frontend-api-guide.md#5-классические-эндпоинты-генерации-справочно)):

```bash
curl -s -X POST "http://10.32.1.46:8200/generate/by_scenario?scenario_id=843&year=2025&source=User&functional_zone_types=residential&functional_zone_types=business&functional_zone_types=industrial" \
  -H "Authorization: Bearer <keycloak_access_token>" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## 5. Ошибки

JSON-RPC коды, см. [app/mcp_server/exceptions.py](../app/mcp_server/exceptions.py):

| Код | Когда |
|---|---|
| `-32002` | `AUTH_TOKEN_EXPIRED` — токен отсутствует/просрочен/отклонён UrbanDB (HTTP 401/403 от апстрима). Взять свежий токен и повторить — **не** ретраить тем же токеном |
| `-32602` | Invalid params — клиентская ошибка (HTTP 4xx от оркестрации: не найден сценарий/зона, невалидная геометрия и т.п.), генерация без `targets_by_zone` и без `use_defaults: true`, невалидные `generation_parameters`, а также `get_generation_result` с некорректным или неизвестным (в т.ч. истёкшим) `result_id` |
| `-32603` | Internal error — серверная ошибка (HTTP 5xx или необработанное исключение), в т.ч. `Failed to load existing buildings` при `preserve_existing_buildings: true` (UrbanDB недоступен) и недоступное хранилище в `get_generation_result` |

---

## 6. Отличие MCP-сервера этого проекта от IDUclub/PzzCompareAPI

В отличие от MCP-сервера PzzCompareAPI (отдельный процесс, ходит в свой API по
HTTP из-за фоновых задач на Celery/Redis), у GenBuilder нет очереди — генерация
выполняется синхронно в рамках запроса. Поэтому MCP-тулзы вызывают слой
оркестрации (`app.logic.generation_orchestration`) напрямую, без лишнего
HTTP round-trip к самому себе — см. [app/mcp_server/server.py](../app/mcp_server/server.py).
