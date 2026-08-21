# GenBuilder API
API for GenBuilder. Can generate images with buildings and other objects for city blocks, vectorize and normalize them.

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
