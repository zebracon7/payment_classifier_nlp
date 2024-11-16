# Payment Classifier

Этот проект предназначен для классификации платежей по категориям на основе их назначения, используя модель BERT.

## Структура проекта

- `predict.py` - скрипт для классификации новых данных.
- `requirements.txt` - список зависимостей.
- `Dockerfile` - файл для создания Docker-образа.
- `run.sh` - скрипт запуска внутри контейнера.
- `data/` - папка с данными.

## Запуск с использованием Docker

1. **Клонировать репозиторий:**

   ```bash
   git clone https://github.com/payment_classifier_nlp
2. **Перейти в директорию проекта:**

    ```bash
    cd payment_classifier_nlp
3. **Заменить файл для классификации:**

Замените data/payments_main.tsv на свой файл с тем же именем и структурой.

4. **Собрать Docker-образ:**

    ```bash
    docker build -t payment_classifier .
5. **Запустить Docker-контейнер:**

    ```bash
    docker run --rm -v "$(pwd)/data:/app/data" payment_classifier
6. **Получить результаты:**
После завершения работы контейнера файл data/classified_payments.tsv будет содержать результаты классификации.