FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
WORKDIR /app
#add pyppi mirror to config
COPY pip.conf /etc/xdg/pip/pip.conf

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV APP_ENV=development

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
