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
> `tools/list` отдаёт все 4 тулзы ниже — идентично локальной Docker-сборке этой же
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

---

## 3. Тулзы

Определены в [app/mcp_server/tools/generation.py](../app/mcp_server/tools/generation.py).

### 3.1. `generate_by_scenario`

Генерация зданий по всей территории сценария UrbanDB.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `scenario_id` | int | ✅ | ID проекта/сценария |
| `year` | int | ✅ | Год данных функциональных зон |
| `source` | string | ✅ | Источник зон, напр. `"OSM"`, `"PZZ"`, `"User"` |
| `functional_zone_types` | list[string] | ✅ | Типы зон для генерации, напр. `["residential", "business", "industrial"]` |
| `physical_object_id` | list[int] | ⛔ | Id физ. объектов, исключить из территории |
| `targets_by_zone` | object | ⛔ | Спрос по зонам (residents / coverage_area / floors_avg / density_scenario / default_floor_group). Без указания — дефолты сервиса |
| `generation_parameters` | object | ⛔ | Низкоуровневые оверрайды генерации (напр. `{"rectangle_finder_step": 5}`) |

**Auth:** нужен bearer-токен.
**Возвращает:** GeoJSON `FeatureCollection` сгенерированных + исключённых зданий
(в `properties` каждой фичи — `floors_count`, `living_area`, `functional_area`,
`building_area`, `zone`, `service`).

### 3.2. `generate_by_territory`

Генерация по присланным полигонам блоков — без привязки к сценарию.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `blocks` | GeoJSON FeatureCollection | ✅ | Polygon-фичи, у каждой заполнен `properties.zone` (напр. `"residential"`) |
| `targets_by_zone` | object | ⛔ | Как выше |
| `generation_parameters` | object | ⛔ | Как выше |

**Auth:** не требуется.
**Возвращает:** GeoJSON `FeatureCollection`.
**Ошибка:** `-32602 Invalid params`, если геометрия блока не Polygon или нет `zone`.

### 3.3. `generate_by_blocks`

Генерация по выбранным functional zone id внутри сценария — свои targets на
каждую зону, один прогон на зону (или на часть полигона, если зона — MultiPolygon).

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `scenario_id`, `year`, `source`, `functional_zone_types` | — | ✅ | Как в `generate_by_scenario` |
| `zones` | list[object] | ✅ | `[{ functional_zone_id, targets_by_zone, generation_parameters? }, ...]` — по записи на зону |
| `physical_object_id` | list[int] | ⛔ | Как выше |

**Auth:** нужен bearer-токен.
**Возвращает:** GeoJSON `FeatureCollection`, объединяющий здания всех запрошенных зон.
**Ошибки:** `-32602`, если `functional_zone_id` не существует для этого
сценария/года/источника, либо геометрия зоны не Polygon/MultiPolygon.

### 3.4. `estimate_max_residents_by_blocks`

Оценка вместимости (макс. число жителей) по зонам на дефолтных
(максимально-плотных) targets — без полного построения застройки.

| Параметр | Тип | Обяз. | Описание |
|---|---|---|---|
| `scenario_id`, `year`, `source`, `functional_zone_types` | — | ✅ | Как выше |
| `functional_zone_ids` | list[int] | ✅ | Зоны для оценки |

**Auth:** нужен bearer-токен. Read-only (`annotations.readOnlyHint: true`).
**Возвращает:** объект `{ <functional_zone_id>: <residents_count> }`.

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
            },
        )

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
        "functional_zone_types": ["residential", "business", "industrial"]
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
| `-32602` | Invalid params — клиентская ошибка (HTTP 4xx от оркестрации: не найден сценарий/зона, невалидная геометрия и т.п.) |
| `-32603` | Internal error — серверная ошибка (HTTP 5xx или необработанное исключение) |

---

## 6. Отличие MCP-сервера этого проекта от IDUclub/PzzCompareAPI

В отличие от MCP-сервера PzzCompareAPI (отдельный процесс, ходит в свой API по
HTTP из-за фоновых задач на Celery/Redis), у GenBuilder нет очереди — генерация
выполняется синхронно в рамках запроса. Поэтому MCP-тулзы вызывают слой
оркестрации (`app.logic.generation_orchestration`) напрямую, без лишнего
HTTP round-trip к самому себе — см. [app/mcp_server/server.py](../app/mcp_server/server.py).
