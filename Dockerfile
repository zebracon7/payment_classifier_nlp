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

# Создаем директорию для модели
RUN mkdir -p data/model

# Скачиваем и распаковываем модель
RUN wget -O model.zip "https://drive.google.com/file/d/1F34WRZNYyn2EI1ZlE1W8PjrhlGrqGarL/view?usp=sharing"
RUN unzip model.zip -d data/model
RUN rm model.zip

# Копируем остальные файлы проекта
COPY . .

# Устанавливаем команду по умолчанию
CMD ["python", "predict.py"]
