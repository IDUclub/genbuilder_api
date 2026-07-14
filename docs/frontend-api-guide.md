# GenBuilder API — гайд для фронтенда

Практическое руководство по интеграции фронта с GenBuilder API. Основной фокус —
новый **разговорный (агентный) режим генерации застройки** поверх SSE; в конце —
краткий справочник по классическим эндпоинтам генерации.

- **Базовый URL:** `<host>` (роутеры подключены без префикса, напр. `https://api.example.com`)
- **Формат:** JSON для классических эндпоинтов; `text/event-stream` (SSE) для чат-режима
- **Версия:** GenBuilder API `0.1.1`

---

## 1. Авторизация

Все эндпоинты требуют **HTTP Bearer**-токен. Теперь это **Keycloak**-токен
пользователя (realm `IDU`); бэкенд проверяет подпись, срок действия и издателя
по JWKS realm-а.

```
Authorization: Bearer <keycloak_access_token>
```

Токен пользователя пробрасывается дальше в UrbanDB. В ChatStorage история пишется
и читается под **сервисным** токеном приложения (client-credentials), а личность
пользователя передаётся заголовком `X-User-Id` (Keycloak `sub`) — так что для
фронта контракт не меняется. Без заголовка — `403 Authorization header missing`;
битый/просроченный/чужой токен — `401 Invalid token`.

---

## 2. Разговорный режим генерации (агент)

### 2.1. Что это

Один эндпоинт, который принимает свободный текстовый запрос («сгенерируй жилую
застройку на 5000 жителей»), сам вытаскивает параметры, при нехватке обязательных
— задаёт уточняющий вопрос (фронт рисует кнопки/инпуты), иначе запускает генерацию
и стримит прогресс → результат → текстовое описание.

```
POST /generate/chat/stream
Content-Type: multipart/form-data
Authorization: Bearer <token>
Accept: text/event-stream
```

### 2.2. Параметры запроса (multipart form-data)

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `user_query` | string | ✅ | Свободный текст запроса пользователя |
| `scenario_id` | int | ⚠️ | ID сценария. Обязателен, **если не** загружается `blocks_file` |
| `year` | int | ⚠️ | Год данных. Обязателен вместе с `scenario_id` |
| `source` | string | ⚠️ | Источник данных (напр. `OSM`). Обязателен вместе с `scenario_id` |
| `blocks_file` | file (GeoJSON) | ⚠️ | Свой набор блоков. Альтернатива сценарию как источник территории |
| `functional_zone_types` | string | ⛔ | CSV-фильтр зон (в агентном режиме игнорируется — всегда residential+business) |
| `chat_id` | string | ⛔ | ID существующего чата для многоходового диалога |
| `project_id` | int | ⛔ | ID проекта (для истории) |
| `model` | string | ⛔ | Переопределить модель LLM |
| `temperature` | float | ⛔ | Переопределить температуру сэмплинга |

**Источник территории — ровно один из двух:**

- **Сценарий** — передай `scenario_id` + `year` + `source`. Зоны и геометрия тянутся из UrbanDB.
- **Файл блоков** — передай `blocks_file` (GeoJSON `FeatureCollection`). См. [раздел 4](#4-загрузка-своих-блоков-blocks_file).

Если не передать ни того, ни другого → `422`. Если передать `scenario_id` без `year`/`source` → `422`.

### 2.3. Обязательный минимум для генерации

Пользователь по сути должен задать **только спрос на жильё**, всё остальное — из
контекста и дефолтов:

- **Территория** — из сценария или файла (не спрашивается у юзера).
- **Спрос по каждой зоне в работе** (`residential` и/или `business`): **число
  жителей (`residents`) ИЛИ жилая площадь в м² (`living_area`)** — они
  взаимозаменяемы (`living_area = residents × la_per_person`).

Этажность, плотность и т.п. — на дефолтах (жилая → 5–8 этажей, многофункциональная
→ 9–16), но если юзер напишет их в тексте («9 этажей, плотность максимум») — будут
учтены.

Если спрос по нужной зоне отсутствует → приходит событие `clarification`, генерация
не запускается.

### 2.4. Как читать ответ (SSE)

Ответ — поток `text/event-stream`. Каждое событие имеет:

- `event:` — **тип** события (см. таблицу ниже);
- `data:` — JSON с полезной нагрузкой (**без** поля `type`).

> ⚠️ Нативный `EventSource` работает только с GET и не умеет слать multipart-тело и
> заголовок `Authorization`. Для этого эндпоинта используй `fetch` + чтение
> `ReadableStream` (пример в [2.7](#27-пример-интеграции-fetch--sse)).

### 2.5. Типы событий

| `event` | Payload (`data`) | Смысл |
|---|---|---|
| `chat_created` | `{ chat_id, title }` | Создан новый чат (если `chat_id` не передавался) |
| `clarification` | `{ content, missing[] }` | Не хватает обязательных параметров — это вопрос, не результат |
| `status` | `{ content, targets_by_zone, functional_zone_types }` | Параметры приняты, генерация стартует |
| `progress` | `{ stage, content }` | Маркер стадии пайплайна |
| `result` | `{ content, summary }` | Готовый `FeatureCollection` + сводка |
| `token` | `{ content }` | Дельта текстового описания результата (стримится по кускам) |
| `warning` | `{ stage, detail, message }` | Некритично (напр. не сохранилось в историю, отброшены объекты файла) |
| `error` | `{ stage, detail }` | Фатально — генерация/ответ не удались |
| `done` | `{ chat_id, assistant_message_id }` | Терминальный маркер потока |

**`clarification.missing[]`** — по элементу на каждую незаполненную зону; готово под рендер контролов:

```json
{
  "content": "Чтобы сгенерировать застройку, уточните:\n— Для зоны «жилая» (residential) укажите спрос на жильё…",
  "missing": [
    {
      "zone": "residential",
      "field": "residents|living_area",
      "control": "number",
      "unit": "чел. или м²",
      "alt_fields": ["residents", "living_area"]
    },
    {
      "zone": "business",
      "field": "residents|living_area",
      "control": "number",
      "unit": "чел. или м²",
      "alt_fields": ["residents", "living_area"]
    }
  ]
}
```

Как рендерить: на каждый элемент `missing` — числовой инпут (`control: "number"`),
подпись из `unit`, а `alt_fields` показывает, что значение можно трактовать как
`residents` **или** `living_area` (можно дать переключатель единиц). Ответ
пользователя отправляется **новым** запросом на тот же эндпоинт с тем же `chat_id`.

**`result`**:

```json
{
  "content": { "type": "FeatureCollection", "features": [ /* здания + исключённые объекты */ ] },
  "summary": {
    "buildings": 128,
    "living_area_total": 350000.0,
    "residents_total": 5000,
    "buildings_by_zone": { "residential": 96, "business": 32 }
  }
}
```

### 2.6. Многоходовой диалог

Состояние держится в истории чата (ChatStorage), не в памяти сервера:

1. Первый запрос без `chat_id` → приходит `chat_created` с новым `chat_id`. Сохрани его.
2. На `clarification` — покажи вопрос/инпуты, дождись ответа пользователя.
3. Отправь ответ (напр. `user_query="5000 жителей"`) с тем же `chat_id`.
4. Сервер склеит прошлые реплики с новой и **заново** разберёт весь диалог, затем
   провалидирует. Хватает минимума → генерит; нет → снова `clarification`.

> Если история недоступна (событие `warning` со `stage: "load_history"`), короткий
> ответ разберётся без контекста — учитывай при UX.

### 2.7. Пример интеграции (fetch + SSE)

```ts
async function generateChat(form: FormData, token: string, onEvent: (type: string, data: any) => void) {
  const res = await fetch("/generate/chat/stream", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, Accept: "text/event-stream" },
    body: form, // FormData с user_query, scenario_id/year/source ЛИБО blocks_file, chat_id, ...
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;

    // события разделены пустой строкой
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      let eventType = "message";
      const dataLines: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) onEvent(eventType, JSON.parse(dataLines.join("\n")));
    }
  }
}
```

Пример использования:

```ts
const form = new FormData();
form.set("user_query", "жилую и многофункциональную застройку на 5000 жителей");
form.set("scenario_id", "198");
form.set("year", "2024");
form.set("source", "OSM");
if (chatId) form.set("chat_id", chatId);

await generateChat(form, token, (type, data) => {
  switch (type) {
    case "chat_created":  chatId = data.chat_id; break;
    case "clarification": renderClarificationInputs(data.missing); break;
    case "status":        showStatus(data.content); break;
    case "progress":      showProgress(data.content); break;
    case "result":        renderBuildings(data.content); showSummary(data.summary); break;
    case "token":         appendAnswerDelta(data.content); break;
    case "warning":       toast(data.message); break;
    case "error":         showError(data.detail); break;
    case "done":          finalize(data.assistant_message_id); break;
  }
});
```

---

## 3. Порядок событий (типичные сценарии)

**Одним запросом (спрос указан сразу):**

```
chat_created → status → progress → result → token* → done
```

**С уточнением:**

```
chat_created → clarification → done          (первый запрос)
status → progress → result → token* → done   (после ответа пользователя, тот же chat_id)
```

`token*` — ноль или более дельт текстового описания.

---

## 4. Загрузка своих блоков (`blocks_file`)

Вместо сценария можно прислать собственный GeoJSON `FeatureCollection`. Требования:

- каждая фича — `Polygon`/`MultiPolygon` с непустым `properties.zone`;
- генерятся только фичи, чья зона нормализуется в **residential** или **business**
  (см. таблицу ниже). Остальные **отбрасываются** — придёт `warning` со
  `stage: "load_blocks"` и числом отброшенных. Если валидных фич нет → `error`.

### Таксономия зон

Имена берутся из `functional_zone_type.name` (UrbanDB). Гранулярные жилые подтипы
сами задают тип застройки:

| `properties.zone` | zone_nickname | → зона генерации | → этажность (floors_group) |
|---|---|---|---|
| `residential` | Жилая | residential | по умолчанию (5–8) |
| `residential_individual` | ИЖС | residential | **private** (ИЖС) |
| `residential_lowrise` | Малоэтажная | residential | **low** (2–4) |
| `residential_midrise` | Среднеэтажная | residential | **medium** (5–8) |
| `residential_multistorey` | Многоэтажная | residential | **high** (9–16) |
| `business` | Общественно-деловая | business | по умолчанию (9–16) |
| `mixed_use` | Многофункциональная | business | по умолчанию (9–16) |
| `unknown`, `basic`, `industrial`, `transport`, `special`, … | — | не генерятся в агентном режиме | — |

> Если у фичи уже проставлен `properties.floors_group`, он побеждает
> производный из подтипа.

Пример минимальной фичи:

```json
{
  "type": "Feature",
  "properties": { "zone": "residential_individual" },
  "geometry": { "type": "Polygon", "coordinates": [ [ [ ... ] ] ] }
}
```

---

## 5. Классические эндпоинты генерации (справочно)

Синхронные, возвращают JSON `FeatureCollection` целиком (без стрима). Полезны, если
фронту нужен прямой вызов без диалога.

### `POST /generate/by_scenario`
Генерация по сценарию. Query: `scenario_id`, `year`, `source`,
`functional_zone_types[]`, `physical_object_id[]` (опц., исключить объекты).
Body (`ScenarioBody`): `targets_by_zone`, `generation_parameters`.
→ `BuildingFeatureCollection`.

### `POST /generate/by_territory`
Генерация по присланным блокам (без сценария). Body (`TerritoryRequest`):
`blocks` (GeoJSON блоков), `targets_by_zone`, `generation_parameters`.
→ `FeatureCollection`.

### `POST /generate/by_blocks`
Генерация по конкретным функциональным зонам сценария. Query: `scenario_id`,
`year`, `source`, `functional_zone_types[]`, `physical_object_id[]`.
Body (`FunctionalZonesRequest`): список `zones` с `functional_zone_id` и
пер-зонными `targets_by_zone` / `generation_parameters`.
→ `FeatureCollection`.

### `POST /generate/max_residents_by_blocks`
Оценка максимального числа жителей по блокам. Query: `scenario_id`, `year`,
`source`, `functional_zone_types[]`, `functional_zone_ids[]`.
→ `{ <functional_zone_id>: <residents> }`.

**Структура `targets_by_zone`** (общая для body классических эндпоинтов):

```json
{
  "residents":        { "residential": 5000, "business": 2000 },
  "floors_avg":       { "residential": 9 },
  "density_scenario": { "residential": "max" },
  "default_floor_group": { "residential": "medium" }
}
```

Нормализация зон из [раздела 4](#таксономия-зон) действует и здесь: гранулярные
жилые подтипы и `mixed_use` обрабатываются во всех эндпоинтах.

---

## 6. Ошибки и статусы

| Код | Когда |
|---|---|
| `403` | Нет/битый `Authorization` заголовок |
| `422` | Нет источника территории (ни `scenario_id`, ни `blocks_file`); `scenario_id` без `year`/`source`; невалидный GeoJSON в `blocks_file` |
| `404` | Не найдены функциональные зоны/сценарий (классические эндпоинты) |
| `503` | LLM-бэкенд не сконфигурирован (`Ollama_API` / `Chat_Model`) — только чат-режим |

В чат-режиме нефатальные проблемы приходят **внутри потока** событием `warning`
(генерация продолжается), фатальные — событием `error` с последующим `done`. HTTP-код
`200` при этом уже отдан (поток открыт), поэтому фронт должен обрабатывать `error`
именно как SSE-событие, а не по статусу ответа.
