FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
#add pyppi mirror to config
COPY pip.conf /etc/xdg/pip/pip.conf

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV APP_ENV=development

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]