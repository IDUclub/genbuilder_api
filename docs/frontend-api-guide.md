# GenBuilder API — гайд для фронтенда

Практическое руководство по интеграции фронта с GenBuilder API. Основной фокус —
новый **разговорный (агентный) режим генерации застройки** поверх SSE; в конце —
краткий справочник по классическим эндпоинтам генерации.

- **Базовый URL:** `<host>` (роутеры подключены без префикса, напр. `https://api.example.com`)
- **Формат:** JSON для классических эндпоинтов; `text/event-stream` (SSE) для чат-режима
- **Версия:** GenBuilder API `0.1.1`

---

## Что изменилось для фронта

Сводка относительно предыдущей версии гайда. Подробности — по ссылкам в разделы.

### Ломает существующую интеграцию

**Режим без проекта (`blocks_file`) теперь всегда задаёт вопрос о существующих
зданиях.** Пока на него не ответили, генерация не запускается — даже если спрос
по всем зонам задан. Приходит обычный `clarification`, но в `missing[]` лежит
элемент с `zone: null`, `field: "existing_buildings"`, `control: "file_or_skip"`,
`optional: true`. Ответить нужно одним из двух полей следующего запроса:

- `buildings_file` — GeoJSON существующих зданий (их пятна вырезаются из кварталов);
- `skip_existing_buildings=true` — пользователь отказался, вопрос больше не задаётся.

Фронт, который этот элемент не обрабатывает, зациклится на `clarification`.
Минимальная правка — всегда слать `skip_existing_buildings=true`; полноценная —
отрисовать выбор файла и кнопку «Нет». См. [4.1](#41-существующие-здания-buildings_file).

В режиме по сценарию поведение прежнее: вопрос не задаётся.

### Новое в SSE-потоке

| Событие | Что даёт | Раздел |
|---|---|---|
| `zones` | Инлайн-подложка функциональных зон **до** генерации — карта рисует её, пока считаются здания | [5.1](#51-событие-zones--подложка) |
| `file` | Дескриптор слоя со ссылкой; та же ссылка ложится в историю чата, так что старый чат можно перерисовать | [5.2](#52-событие-file--дескриптор-слоя) |

Оба события новые: раньше поток шёл `chat_created → status → progress → result → token* → done`.
Актуальный порядок и сырой пример потока — в [3.1](#31-сырой-поток-пример).

`chat_created.title` теперь составляет LLM (3–5 слов по первому сообщению) —
раньше это был обрезанный текст запроса. Показывай как есть, свою обрезку убери.

### Новые эндпоинты

| Эндпоинт | Зачем | Раздел |
|---|---|---|
| `GET /layers/functional_zones` | Живой GeoJSON зон сценария по ссылке из события `file` | [5.3](#53-get-layersfunctional_zones) |
| `GET /files/{slot}/{result_id}` | Забрать сохранённый слой: `buildings`, `blocks_input`, `existing_buildings` (30 дней) | [5.4](#54-get-filesslotresult_id) |
| `GET /generate/properties_schema` | Русские подписи свойств зданий и значений enum-полей; константа, кэшируется | [7.3](#73-get-generateproperties_schema) |

Оба геослойных эндпоинта требуют `Authorization`, поэтому `<a href>` и
`window.open` для скачивания не годятся — как качать файлом, см. [5.5](#55-как-скачать-слой-файлом).

### Изменения в существующих эндпоинтах

- `POST /generate/chat/stream` — новые поля формы `buildings_file` и
  `skip_existing_buildings` (см. выше).
- `POST /generate/by_territory` — в теле появилось поле `existing_buildings`
  (GeoJSON исключаемых зданий), а `blocks` теперь принимают не только `Polygon`,
  но и `MultiPolygon`. См. [раздел 6](#6-классические-эндпоинты-генерации-справочно).
- Ответ генерации — в коллекцию попадают исключённые существующие объекты с
  `is_excluded: true`. У сгенерированных зданий этого ключа **нет вообще**,
  проверять нужно `feature.properties.is_excluded === true`.
  См. [7.2](#72-как-отличить-исключённые-объекты).

### Контракта не касается

- LLM-бэкенд переехал с Ollama на vLLM. Меняются только допустимые значения
  необязательного поля `model`: теперь это имя модели, обслуживаемой vLLM. Если
  фронт его не передаёт (рекомендуется), делать ничего не нужно.
- Появились MCP-инструменты, A2A-агент и админский неймспейс `/admin/config` —
  он под отдельным админским доступом и во фронтовом контракте не участвует.

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
| `buildings_file` | file (GeoJSON) | ⛔ | Существующие здания — их пятна исключаются из генерации (режим без проекта) |
| `skip_existing_buildings` | bool | ⛔ | `true`, если пользователь отказался загружать существующие здания |
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

**Режим без проекта (`blocks_file` без `scenario_id`) спрашивает ещё об одном.**
У сценария существующая застройка берётся из UrbanDB, а у своего файла кварталов
её взять неоткуда — поэтому перед первой генерацией бэкенд один раз задаёт
вопрос: **«Хотите загрузить существующие здания?»** Он приходит тем же событием
`clarification`, отдельным элементом `missing[]` с `field: "existing_buildings"` и
`optional: true`. Ответ — новый запрос с тем же `chat_id` и:

- `buildings_file` — GeoJSON существующих зданий: их пятна вырезаются из
  кварталов, генерация обходит существующую застройку, а сами здания приходят в
  `result` с `is_excluded: true` (см. [7.2](#72-как-отличить-исключённые-объекты));
- `skip_existing_buildings=true` — пользователь отказался, застройка генерируется
  по всей площади кварталов.

Пока не пришло ни файла, ни отказа, вопрос повторяется на каждом запросе. Если
фронт знает ответ заранее (пользователь ответил до отправки), можно приложить
`buildings_file` / `skip_existing_buildings` сразу к первому запросу — тогда
лишнего круга не будет. Подробности — в [4.1](#41-существующие-здания-buildings_file).

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
| `chat_created` | `{ chat_id, title }` | Создан новый чат (если `chat_id` не передавался). `title` — короткий заголовок, который LLM составляет по первому сообщению; показывай его в списке чатов как есть |
| `clarification` | `{ content, missing[] }` | Не хватает обязательных параметров — это вопрос, не результат |
| `status` | `{ content, targets_by_zone, functional_zone_types }` | Параметры приняты, генерация стартует |
| `progress` | `{ stage, content }` | Маркер стадии пайплайна |
| `zones` | `{ source, content }` | Подложка функциональных зон, инлайн, **до** генерации (см. [5.1](#51-событие-zones--подложка)) |
| `result` | `{ content, summary }` | Готовый `FeatureCollection` + сводка |
| `file` | дескриптор слоя | Ссылка на слой; она же ложится в историю чата (см. [5.2](#52-событие-file--дескриптор-слоя)) |
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
      "alt_fields": ["residents", "living_area"],
      "optional": false
    },
    {
      "zone": null,
      "field": "existing_buildings",
      "control": "file_or_skip",
      "unit": null,
      "alt_fields": ["buildings_file", "skip_existing_buildings"],
      "optional": true
    }
  ]
}
```

Как рендерить:

- `control: "number"` — числовой инпут, подпись из `unit`; `alt_fields`
  показывает, что значение можно трактовать как `residents` **или**
  `living_area` (можно дать переключатель единиц);
- `control: "file_or_skip"` — вопрос с двумя ответами: выбор файла (уйдёт полем
  `buildings_file`) и кнопка «Нет» (уйдёт как `skip_existing_buildings=true`).
  Приходит только в режиме без проекта, `zone` у него `null`;
- `optional: true` — на этот вопрос можно ответить отказом, но ответить нужно:
  пока ответа нет, генерация не запускается.

Ответ пользователя отправляется **новым** запросом на тот же эндпоинт с тем же
`chat_id`. Все элементы `missing` приходят одним событием — их можно показать
одной формой и ответить одним запросом.

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

**`zones`** и **`file`** описаны в [разделе 5](#5-геослои-подложка-и-ссылки-в-истории).

### 2.6. Многоходовой диалог

Состояние держится в истории чата (ChatStorage), не в памяти сервера:

1. Первый запрос без `chat_id` → приходит `chat_created` с новым `chat_id`. Сохрани его.
   Там же `title` — заголовок чата (3–5 слов, составлен LLM по первому сообщению;
   если LLM недоступна, это просто обрезанный текст запроса). Заголовок задаётся
   **один раз**, при создании чата, и на последующих ходах не меняется.
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

**Одним запросом, по сценарию (спрос указан сразу):**

```
chat_created → status → zones → file(functional_zones) → progress
             → result → file(buildings) → token* → done
```

**Одним запросом, со своим файлом кварталов** (вопрос про существующие здания уже
отвечен — приложен `buildings_file` либо `skip_existing_buildings=true`):

```
chat_created → status → zones → progress
             → result → file(buildings) → file(blocks_input)
             → file(existing_buildings)? → token* → done
```

**Режим без проекта, вопрос про существующие здания:**

```
chat_created → clarification → done                    (первый запрос)
status → zones → … → result → token* → done            (после ответа, тот же chat_id)
```

**С уточнением:**

```
chat_created → clarification → done          (первый запрос)
status → zones → … → result → token* → done  (после ответа пользователя, тот же chat_id)
```

`token*` — ноль или более дельт текстового описания.

> События `file` могут и не прийти: если объектное хранилище недоступно, вместо
> них будет `warning` со `stage: "store_layer"`, а поток всё равно дойдёт до
> `done` — результат в событии `result` от этого не страдает, теряется только
> возможность забрать слой позже.

### 3.1. Сырой поток (пример)

Успешная генерация по сценарию. Большие `FeatureCollection` подрезаны, остальное
приведено как есть.

```
event: chat_created
data: {"chat_id": "9f3a…", "title": "Построй жильё на 5000 человек"}

event: status
data: {"content": "Параметры приняты, запускаю генерацию застройки.", "targets_by_zone": {"residents": {"residential": 5000, "business": 2000}, "default_floor_group": {"residential": "medium", "business": "high"}}, "functional_zone_types": ["residential", "business"]}

event: zones
data: {"source": "scenario", "content": {"type": "FeatureCollection", "features": [ /* полигоны зон */ ]}}

event: file
data: {"name": "functional_zones", "title": "Функциональные зоны", "role": "input", "url": "http://10.32.1.46:8200/layers/functional_zones?scenario_id=198&year=2024&source=OSM&functional_zone_types=residential&functional_zone_types=business", "download_url": null, "filename": "functional_zones.geojson", "mime_type": "application/geo+json", "source_service": "genbuilder"}

event: progress
data: {"stage": "generation", "content": "Генерация зданий…"}

event: result
data: {"content": {"type": "FeatureCollection", "features": [ /* здания */ ]}, "summary": {"buildings": 128, "living_area_total": 350000.0, "residents_total": 5000, "buildings_by_zone": {"residential": 96, "business": 32}}}

event: file
data: {"name": "buildings", "title": "Сгенерированная застройка", "role": "result", "url": "http://10.32.1.46:8200/files/buildings/9fb46d53957b4e459a77dbe018dc96d2", "download_url": null, "filename": "buildings.geojson", "mime_type": "application/geo+json", "source_service": "genbuilder"}

event: token
data: {"content": "Сгенерирована застройка "}

event: token
data: {"content": "на 5000 жителей."}

event: done
data: {"chat_id": "9f3a…", "assistant_message_id": "b71c…"}
```

Что из этого следует для реализации:

- **Тип события лежит в поле `event`, а не внутри `data`.** Сервер вынимает
  `type` из конверта и делает его именем SSE-события; в `data` остаётся всё
  остальное. Диспатчить нужно по `event`.
- **Два события `file` с разной природой.** Различай по `name` (или `role`):
  `functional_zones` — живой запрос, работает всегда; `buildings` — файл из
  хранилища, живёт 30 дней. Форма одинаковая, взаимозаменяемыми они не являются.
- **Между `zones` и `result` проходит всё время генерации** — в этом и смысл
  раннего `zones`.
- **`token` дробится произвольно**, как отдал LLM; склеивать на стороне клиента.
- **`done` — терминатор, а не носитель результата.** Полезная нагрузка пришла
  раньше; из него берут только `chat_id` и `assistant_message_id`.

В режиме `blocks_file` первого `file` (`functional_zones`) не будет, зато после
`file(buildings)` придёт второй — `blocks_input`.

Ошибка в потоке выглядит так и приходит **вместо** `result`:

```
event: error
data: {"stage": "generation", "detail": "…"}

event: done
data: {"chat_id": "9f3a…", "assistant_message_id": null}
```

Обрабатывать её нужно именно как событие: HTTP `200` к этому моменту уже отдан,
поток открыт, и по статусу ответа о сбое узнать нельзя.

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

> **Расхождение, о котором нужно знать.** В историю чата (слой `blocks_input`)
> кладётся файл **ровно как ты его загрузил**, вместе с отброшенными фичами.
> Инлайн-событие `zones` при этом уже отфильтровано. Это сделано намеренно: файл
> пользователя мы не переписываем, а подложка обязана совпадать с тем, из чего
> реально сгенерированы здания.

### 4.1. Существующие здания (`buildings_file`)

Своим файлом кварталов территория задаётся «с нуля», и генератор по умолчанию
считает её пустой. Чтобы застройка не выросла поверх того, что уже стоит,
приложи существующие здания:

- `buildings_file` — GeoJSON `FeatureCollection`, фичи `Polygon`/`MultiPolygon`;
- `properties` **необязательны**: для исключения достаточно геометрии. Если они
  есть, из них берутся `floors_count`, `living_area`, `building_area`,
  `residents_number`, `building_type`, `zone`, `service` — с ними здание
  отрисуется как обычный объект;
- неполигональные фичи отбрасываются: придёт `warning` со
  `stage: "load_existing_buildings"` и числом отброшенных. Если полигонов нет
  вовсе — тоже `warning`, но генерация продолжится (без исключений), а не упадёт.

> **Отбора на бэкенде нет: отбор — это сам файл.** Исключается **каждая**
> полигональная фича из `buildings_file`; никакого флага в `properties` для
> этого нет. Это осознанное отличие от режима по сценарию, где в UrbanDB лежат
> все физобъекты территории и `physical_object_id[]` выбирает из них
> подмножество. Здесь «всех зданий» у бэкенда нет, поэтому подмножество
> формирует фронт: если пользователь отмечает часть зданий на карте — собери
> `FeatureCollection` только из отмеченных и отправь его.

Что делает бэкенд: пятна зданий (с отступом, подобранным по размеру здания)
вырезаются из кварталов **до** генерации, а сами здания возвращаются в `result`
вместе со сгенерированными — с `is_excluded: true`. То есть на карте это один
слой: новая застройка плюс существующая, различимые по флагу.

Тот же механизм в классическом режиме — поле `existing_buildings` в теле
`POST /generate/by_territory` (см. [раздел 6](#6-классические-эндпоинты-генерации-справочно)).

---

## 5. Геослои: подложка и ссылки в истории

Кроме инлайн-результата чат отдаёт **слои**: подложку функциональных зон и
ссылки, по которым слой можно забрать позже — в том числе при открытии старого
чата, когда потока уже нет.

Принцип разделения: **что сгенерировали сами — храним у себя, что принадлежит
другому сервису — отдаём ссылкой на него**.

| Слой (`name`) | Откуда | Ссылка | Живёт |
|---|---|---|---|
| `buildings` | наш результат генерации | `/files/buildings/{result_id}` | 30 дней |
| `blocks_input` | файл пользователя, как загружен | `/files/blocks_input/{result_id}` | 30 дней |
| `existing_buildings` | полигоны из файла существующих зданий | `/files/existing_buildings/{result_id}` | 30 дней |
| `functional_zones` | живой запрос в UrbanDB | `/layers/functional_zones?…` | бессрочно |

### 5.1. Событие `zones` — подложка

Приходит **до** `progress`, то есть до того как посчитаны здания: карту можно
отрисовать сразу, не дожидаясь результата.

```json
{
  "source": "scenario",
  "content": { "type": "FeatureCollection", "features": [ /* полигоны зон */ ] }
}
```

- `source: "scenario"` — зоны сценария из UrbanDB, нормализованные и
  отфильтрованные ровно так, как их видит генерация;
- `source: "blocks_file"` — отфильтрованные кварталы из загруженного файла
  (только residential/business), то есть то, что реально ушло в генерацию.

Если зоны подтянуть не удалось, придёт `warning` со `stage: "zones"`. Генерация
при этом идёт своим ходом — не будет только подложки.

### 5.2. Событие `file` — дескриптор слоя

```json
{
  "name": "buildings",
  "title": "Сгенерированная застройка",
  "role": "result",
  "url": "http://10.32.1.46:8200/files/buildings/6f1c…e2",
  "download_url": null,
  "filename": "buildings.geojson",
  "mime_type": "application/geo+json",
  "source_service": "genbuilder"
}
```

- `role` — `result` для сгенерированного, `input` для исходных данных;
- `download_url` **всегда `null`**: объектное хранилище живёт в приватной сети,
  браузер туда не ходит, весь трафик идёт через GenBuilder по `url`;
- `url` абсолютный, если на сервере задан `PUBLIC_BASE_URL`, иначе относительный.

Та же нагрузка (без `download_url` и `role`) сохраняется в историю чата как
часть сообщения ассистента:

```json
{
  "role": "assistant",
  "parts": [
    { "kind": "text", "payload": { "text": "Сгенерировано 128 зданий…" } },
    { "kind": "file", "payload": { "url": "…/layers/functional_zones?scenario_id=198&…", "name": "functional_zones", … } },
    { "kind": "file", "payload": { "url": "…/files/buildings/6f1c…e2", "name": "buildings", … } }
  ]
}
```

При отрисовке истории: текст берётся из `text`-части, слои — из `file`-частей по
их `url`. Отдельного «текстового» поля `content` у таких сообщений нет.

### 5.3. `GET /layers/functional_zones`

Функциональные зоны сценария — **живой** запрос, ничего не копируется.

Query: `scenario_id`, `year`, `source`, `functional_zone_types[]` (повторяемый,
опционален — без него вернутся все зоны).

→ `FeatureCollection` в EPSG:4326. Зоны нормализованы так же, как их видит
генерация (гранулярные жилые подтипы схлопнуты в `residential`, `mixed_use` — в
`business`), у фич сохраняется `functional_zone_id` для связывания.

Требует токен пользователя: доступ к приватному сценарию проверяет UrbanDB на
**каждый** запрос. Поэтому ссылка не устаревает и не является «капабилити» —
отозвали доступ к сценарию, и по ней ничего не отдастся.

### 5.4. `GET /files/{slot}/{result_id}`

Наши собственные артефакты из объектного хранилища. `slot` — `buildings`,
`blocks_input` или `existing_buildings`.

→ `application/geo+json`, тело стримится чанками, `Content-Disposition:
attachment`.

Требует токен. Ответы: `404` — неизвестный слот, битый `result_id` **или**
истёкший объект.

> **30 дней.** Слои `buildings`, `blocks_input` и `existing_buildings` удаляются из хранилища по
> lifecycle-правилу через 30 дней. Открытие старого чата — штатная ситуация, в
> которой ссылка отдаст `404`: показывай слой как недоступный и **не роняй**
> просмотр истории. Ссылка на `functional_zones` при этом продолжает работать —
> она не про хранилище.

### 5.5. Как скачать слой файлом

`<a href>` и `window.open` не годятся: заголовок `Authorization` туда не
поставить. Нужен `fetch` → `Blob` → `createObjectURL`:

```ts
async function downloadLayer(layer: { url: string; filename: string }, token: string) {
  const res = await fetch(layer.url, { headers: { Authorization: `Bearer ${token}` } });
  if (res.status === 404) {
    throw new Error("Слой больше недоступен: результаты генерации хранятся 30 дней");
  }
  if (!res.ok) throw new Error(`Не удалось скачать слой: ${res.status}`);

  const blob = await res.blob();
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = layer.filename;
  a.click();
  URL.revokeObjectURL(href);
}
```

Для отрисовки на карте вместо `blob()` бери `await res.json()` — это готовый
`FeatureCollection`.

---

## 6. Классические эндпоинты генерации (справочно)

Синхронные, возвращают JSON `FeatureCollection` целиком (без стрима). Полезны, если
фронту нужен прямой вызов без диалога.

### `POST /generate/by_scenario`
Генерация по сценарию. Query: `scenario_id`, `year`, `source`,
`functional_zone_types[]`, `physical_object_id[]` (опц., исключить объекты).
Body (`ScenarioBody`): `targets_by_zone`, `generation_parameters`.
→ `BuildingFeatureCollection`.

### `POST /generate/by_territory`
Генерация по присланным блокам (без сценария). Body (`TerritoryRequest`):
`blocks` (GeoJSON блоков), `existing_buildings` (опц., GeoJSON существующих
зданий — исключаются из генерации), `targets_by_zone`, `generation_parameters`.
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

## 7. Свойства зданий в ответе

### 7.1. Состав `properties`

У каждой сгенерированной постройки в `properties` приходит восемь полей:

| Поле | Тип | Смысл |
|---|---|---|
| `floors_count` | number | Этажность |
| `living_area` | number | Жилая площадь, м² (0 для нежилых зон) |
| `building_area` | number | Общая площадь здания, м² |
| `residents_number` | number | Расчётное число жителей |
| `building_type` | enum | Тип застройки (`private`, `low`, `medium`, `high`, …) |
| `zone` | enum | Функциональная зона блока (нормализованная) |
| `service` | array | Сервисы в здании (может быть пустым) |
| `broke_restriction_zone` | boolean | Нарушены нормативные отступы |

### 7.2. Как отличить исключённые объекты

Если в `/generate/by_scenario` или `/generate/by_blocks` передан
`physical_object_id[]`, в ту же коллекцию попадают **существующие** объекты,
исключённые из генерации. У них те же восемь полей плюс два дополнительных:

| Поле | Тип | Смысл |
|---|---|---|
| `is_excluded` | boolean | Всегда `true` — признак существующего объекта |
| `physical_object_id` | integer | ID физического объекта в UrbanDB |

> ⚠️ У сгенерированных зданий ключа `is_excluded` **нет вообще** — он не
> приходит со значением `false`. Проверять нужно наличие или истинность:
> `feature.properties.is_excluded === true`.

В `/generate/by_territory` параметра `physical_object_id` нет — там существующие
здания приходят телом запроса (`existing_buildings`), а в чат-режиме без проекта
файлом `buildings_file`. Помечаются они так же (`is_excluded: true`), но
`physical_object_id` у них `null`, если его не было в присланных `properties`.

### 7.3. `GET /generate/properties_schema`

Отдаёт русские подписи для имён свойств и для значений enum-полей. Авторизация
не требуется, ответ константный — запрашивайте один раз и кэшируйте.

```json
{
  "properties": {
    "floors_count": { "label": "Количество этажей", "kind": "number", "unit": "эт.", "excluded_only": false },
    "is_excluded":  { "label": "Существующий объект", "kind": "boolean", "unit": null, "excluded_only": true }
  },
  "values": {
    "building_type": { "private": "ИЖС", "medium": "Среднеэтажная" },
    "zone": { "residential": "Жилая", "business": "Общественно-деловая" }
  }
}
```

- `kind` — как рендерить значение: `number`, `integer`, `boolean`, `enum`, `array`.
- `unit` — единица измерения либо `null`.
- `excluded_only` — поле есть только у исключённых объектов.
- `values` — словари подписей для полей с `kind: "enum"`.

---

## 8. Ошибки и статусы

| Код | Когда |
|---|---|
| `403` | Нет/битый `Authorization` заголовок |
| `422` | Нет источника территории (ни `scenario_id`, ни `blocks_file`); `scenario_id` без `year`/`source`; невалидный GeoJSON в `blocks_file` или `buildings_file` |
| `404` | Не найдены функциональные зоны/сценарий (классические эндпоинты); слой недоступен или истёк (`/files/{slot}/{result_id}`) |
| `401` | Битый/просроченный токен на `/layers/functional_zones` и `/files/{slot}/{result_id}` |
| `503` | LLM-бэкенд не сконфигурирован (`LLM_API` / `Chat_Model`) — только чат-режим |

В чат-режиме нефатальные проблемы приходят **внутри потока** событием `warning`
(генерация продолжается), фатальные — событием `error` с последующим `done`. HTTP-код
`200` при этом уже отдан (поток открыт), поэтому фронт должен обрабатывать `error`
именно как SSE-событие, а не по статусу ответа.
