# GenBuilder API
API for GenBuilder. Can generate images with buildings and other objects for city blocks, vectorize and normalize them.

## 3D facade jobs

The asynchronous 3D endpoints require a separate `facade-jobs` deployment:

| Environment variable | Required | Default | Purpose |
|---|---:|---:|---|
| `FACADE_JOBS_API` | yes | — | Internal base URL used by GenBuilder, for example `http://facade-jobs:8000` |
| `FACADE_JOBS_PUBLIC_API` | no | `FACADE_JOBS_API` | Public base URL used in the returned `status_url` |
| `FACADE_JOBS_TIMEOUT_SECONDS` | no | `30` | Timeout for queue submission; this does not cover facade generation time |

See [the frontend API guide](docs/frontend-api-guide.md#3d-фасады) and
[the integration plan](docs/facades-3d-integration-plan.md) for the endpoint and
service contracts.

All `/generate/3d/*` endpoints accept an optional Russian preset name or English
prompt in the `facade_style` query parameter. `/generate/chat/stream/3d`
accepts the same value as a multipart field and can also extract arbitrary
Russian descriptions from `user_query`. Omitting the style picks the default
preset of each functional zone.

### Facade library

Built-in styles can also be assembled synchronously from pre-generated facade
sections stored in MinIO under `FACADE_LIBRARY_PREFIX`. Each style has three
seed variants per width (6, 9, 12, 18, 24 m), each generated as an 8-floor, 32 m wall
because Facades-3D draws one floor per about 4 m of wall height. The section is
cut into ground, typical and top floors, and every wall is stacked from them to
the building's real floor count; walls wider than 24 m are split into equal
segments. A building picks one variant by a hash of its id, so all its walls
match while neighbouring buildings differ. `facade-jobs` still reads the older `facade-library/v1` layout. The
`facade_source` query parameter of `/generate/3d/*` selects `gpu` (queue a
job, `202`), `library` (assemble now, `200`, nearest width on a miss) or
`library_then_gpu` (library when every wall has a section, otherwise a job). Library scenes are served by
`GET /facade-scenes/{result_id}.glb`; style previews by `GET /facade-styles` and
`GET /facade-styles/{style_id}/preview.glb`.

| Environment variable | Default | Purpose |
|---|---:|---|
| `FACADE_SOURCE_DEFAULT` | `gpu` | Source used when `facade_source` is omitted |
| `FACADE_LIBRARY_PREFIX` | `facade-library/v2` | Object prefix of the library manifest, sections and previews |
| `FACADE_LIBRARY_PPM` | `32` | Texture resolution of the sections to use |
| `FACADE_LIBRARY_MAX_WIDTH_SCALE` | `2.5` | Largest horizontal stretch of a section |
| `FACADE_LIBRARY_MAX_WALLS` | `500` | Larger scenes go to `facade-jobs` (or `413` without it) |
| `FACADE_LIBRARY_MANIFEST_TTL_SECONDS` | `300` | How long the manifest and preview index are cached |

The library and the previews are filled by
[`scripts/build_facade_library.py`](scripts/build_facade_library.py) from a
machine that reaches both MinIO and the Facades-3D GPU host (VPN). It reads the
`FILESERVER_*` variables and ignores the local HTTP proxy:

```bash
python scripts/build_facade_library.py check
python scripts/build_facade_library.py prewarm --dry-run
python scripts/build_facade_library.py prewarm --styles brick glass
python scripts/build_facade_library.py previews
```

`prewarm` generates `--variants` sections (3 by default) per style and width
(`--section-floors` and `--section-floor-height`, 8 x 4 m by default), skips sections that already exist
(`--force` regenerates them) and rewrites the manifest after every section, so
it can be interrupted and resumed. `previews` stacks 3, 6 and 12 floors of each
style's 12 m variant-0 section into the low, medium and high preview boxes.

## Configuration

The deployment workflow can build `.env.development` directly from GitHub
repository settings (`Settings` → `Secrets and variables` → `Actions`):

- variable `ENV_APP` — non-sensitive application parameters;
- variable `ENV_URLS` — service URLs, hosts and ports;
- secret `ENV_SECRET` — credentials and `ADMIN_API_TOKEN`.

Each value is a multiline dotenv fragment. They are concatenated in that order,
so `ENV_SECRET` wins if a key is duplicated. `ENV_FILE` and `ENV_PATH` remain as
legacy fallbacks. See [`.env.example`](.env.example) for the supported keys and
their recommended grouping.

### Runtime configuration API

With `ADMIN_API_TOKEN` set, effective non-sensitive settings can be changed
without rebuilding or restarting the container. Send the token in the
`X-Admin-Token` header:

- `GET /admin/config/settings` — typed effective settings (secrets masked);
- `GET /admin/config/overrides` — active persistent overrides;
- `GET /admin/config/{key}` — effective value and override status;
- `PUT /admin/config/{key}` with `{"value":"...","updated_by":"..."}` — validate and set;
- `DELETE /admin/config/{key}` — remove and restore the deployed value;
- `POST /admin/config/reload` — force this process to sync immediately.

Example:

```bash
curl -X PUT http://localhost:8200/admin/config/LLM_API \
  -H 'X-Admin-Token: <ADMIN_API_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{"value":"http://new-llm:8001","updated_by":"operator"}'
```

Overrides are stored in SQLite at `RUNTIME_CONFIG_PATH` on the persistent
`runtime_config` Docker volume and synced by every process (5-second TTL by
default). Unknown keys, credentials and boot-only settings are rejected.

## Generated geo layers

Chat generation stores its own artefacts (the generated buildings and, when the
user uploads one, the blocks file) in S3-compatible object storage and hands the
frontend durable links to them. Functional zones are **not** stored: they belong
to the scenario in UrbanDB and are served as a live query, so access is
re-checked on every fetch.

| Environment variable | Required | Default | Purpose |
|---|---:|---:|---|
| `FILESERVER_ENDPOINT` | yes* | — | MinIO host:port of the S3 API, for example `10.32.1.42:9000` |
| `FILESERVER_ACCESS_KEY` | yes* | — | Scoped access key |
| `FILESERVER_SECRET_KEY` | yes* | — | Scoped secret key |
| `FILESERVER_BUCKET_NAME` | yes* | — | Bucket holding the layers, for example `genbuilder` |
| `FILESERVER_SECURE` | no | `false` | Use HTTPS towards MinIO |
| `FILESERVER_REGION` | no | `us-east-1` | Sent explicitly so the client never calls `GetBucketLocation`, a right the scoped credentials do not have |
| `OUTPUTS_DIR` | no | `outputs` | Local fallback directory, used only when no `FILESERVER_*` variable is set |
| `DEFAULT_SERVICES_TERRITORY_ID` | no | `1` | UrbanDB region whose service normatives place services in the project-less chat mode when the request has neither `territory_id` nor `project_id`. `1` is Leningrad Oblast; an empty value disables the fallback |

\* All four are required together. A partial set is refused at startup of the
storage backend rather than silently degraded to local disk — otherwise
production links would break at the next container restart.

The bucket is provisioned out of band; the credentials carry no `CreateBucket`
right and nothing is created at runtime.

**Retention is a bucket rule, not code.** Set a 30-day expiry on the bucket, or
stored layers accumulate forever:

```bash
mc ilm rule add --expire-days 30 <alias>/genbuilder
```

The frontend must treat a `404` from `/files/{slot}/{result_id}` as an expected
outcome when reading an old chat. See
[the frontend API guide](docs/frontend-api-guide.md#5-геослои-подложка-и-ссылки-в-истории).
