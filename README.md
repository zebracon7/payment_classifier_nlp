
# Автоматизация разбора платежей на основании назначения платежа

## Описание проекта

Этот проект разработан для автоматической классификации назначений платежей на основании назначения платежа. Основная задача — обучить модель, способную с высокой точностью определять категорию платежа. Проект реализован в рамках хакатона BIV Hack Challenge с использованием предобученной модели **BERT** для обработки текстов на русском языке.

## Постановка задачи

**Цель:**  
Создать модель, которая классифицирует текстовые данные назначений платежей по заранее определенным категориям. 

**Категории:**  
- BANK_SERVICE
- FOOD_GOODS
- NON_FOOD_GOODS
- LEASING
- LOAN
- REAL_ESTATE
- SERVICE
- TAX
- NOT_CLASSIFIED

**Этапы решения:**
1. **Предобработка данных:**
   - Расширение датасета
   - Удаление специальных символов, чисел и лишних пробелов.
   - Приведение текста к нижнему регистру.
   - Токенизация текста.
   - Обработка дисбаланса данных для равномерного распределения категорий.
   
2. **Обучение модели:**
   - Использование предобученной модели BERT на русском языке от DeepPavlov.
   - Настройка гиперпараметров, включая размер батча, количество эпох и скорость обучения.
   - Дообучение модели на предоставленных данных.

3. **Классификация новых данных:**
   - Применение обученной модели к новым текстам для определения категории.

## Реализация

Модель была реализована с использованием предобученной модели `DeepPavlov/rubert-base-cased`. Мы провели дообучение (fine-tuning), чтобы адаптировать модель под задачу классификации текстов назначений платежей. 

**Архитектура модели:**
1. **Входные данные:**
   - Текстовые назначения платежей после предобработки.

2. **Токенизация:**
   - Используется токенизатор BERT из предобученной модели `DeepPavlov/rubert-base-cased`.
   - Преобразует текст в последовательность токенов и соответствующих им идентификаторов.

3. **Модель BERT:**
   - Энкодер BERT состоит из 12 слоев трансформеров.
   - Использует механизмы самовнимания (*self-attention*) для учета контекста текста.
   - Предобученные веса модели адаптированы к особенностям русского языка.

4. **Полносвязный слой классификации:**
   - Выходной вектор `[CLS]` из BERT подается на полносвязный слой.
   - Активация: Используется функция *Softmax* для получения вероятностей классов.

5. **Выходные данные:**
   - Категория платежа, предсказанная на основе анализа текста.

## Результаты
Модель демонстрирует высокую точность на тестовом датасете. Значение Accuracy = 98%

## Структура проекта

- `predict.py` - скрипт для классификации новых данных.
- `requirements.txt` - список зависимостей.
- `Dockerfile` - файл для создания Docker-образа.
- `data/` - папка с данными.

## Запуск с использованием Docker

1. **Клонировать репозиторий:**

   ```bash
   git clone https://github.com/zebracon7/payment_classifier_nlp
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

## Выводы и улучшения

### Достигнутые результаты:
- Высокая точность классификации по большинству категорий.
- Модель успешно справляется с текстами, содержащими явные ключевые слова.

### Предложения по улучшению:
1. Расширить тренировочный датасет, особенно для редких категорий.
2. Использовать методы увеличения данных (Data Augmentation), такие как перефразирование и добавление синонимов.
3. Ансамблировать предсказания нескольких моделей для повышения точности.
4. Интегрировать внешние знания (финансовые онтологии или словари терминов).