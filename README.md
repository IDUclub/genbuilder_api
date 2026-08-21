# GenBuilder API
API for GenBuilder. Can generate images with buildings and other objects for city blocks, vectorize and normalize them.

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
| `PUBLIC_BASE_URL` | no | — | Absolute base for the links written into chat history, for example `http://10.32.1.46:8200`. Without it links are relative, which breaks history read from another origin |

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
