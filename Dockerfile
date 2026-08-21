FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
#add pyppi mirror to config
COPY pip.conf /etc/xdg/pip/pip.conf

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV APP_ENV=development
ENV RUNTIME_CONFIG_PATH=/runtime-config/overrides.sqlite3

# iduconfig requires the APP_ENV file to exist. Runtime values are injected by
# Compose; keep the image-side file empty so deployment secrets are not baked in.
RUN touch .env.development

COPY . .

EXPOSE 8000

EXPOSE 9464

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
