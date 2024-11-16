# Используем официальный образ Python 3.9
FROM python:3.9-slim

# Устанавливаем необходимые системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем переменную окружения для отключения параллелизма токенизатора
ENV TOKENIZERS_PARALLELISM=false

# Создаем рабочую директорию
WORKDIR /app

# Копируем только необходимые файлы для установки зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем gdown для загрузки файлов с Google Drive
RUN pip install gdown

# Копируем остальные файлы проекта
COPY . .

# Устанавливаем команду по умолчанию
CMD sh -c "\
    gdown --id 1F34WRZNYyn2EI1ZlE1W8PjrhlGrqGarL -O model.zip && \
    unzip model.zip -d data/ && \
    rm model.zip && \
    python predict.py \
"
