import os
# Отключаем параллелизм токенизатора
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Остальные импорты
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


def main():
    """
    Основная функция для обучения модели и сохранения ее весов.
    """
    # Путь к папке с данными
    data_path = 'data'
    model_save_path = os.path.join(data_path, 'model')

    # Проверяем наличие необходимых файлов
    check_files(data_path, ['payments_training.tsv'])

    # Загрузка данных
    training_data = load_training_data(os.path.join(data_path, 'payments_training.tsv'))

    # Предобработка текста
    training_data['text'] = training_data['Назначение платежа'].apply(preprocess_text)

    # Кодирование меток
    label_encoder = LabelEncoder()
    training_data['label'] = label_encoder.fit_transform(training_data['Категория'])

    # Сохранение LabelEncoder для последующего использования
    import pickle
    with open(os.path.join(model_save_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(
        training_data['text'],
        training_data['label'],
        test_size=0.2,
        random_state=42
    )

    # Загрузка модели и токенизатора
    model_name = 'DeepPavlov/rubert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )

    # Создание датасетов и загрузчиков данных
    train_dataset = PaymentDataset(X_train, y_train, tokenizer)
    val_dataset = PaymentDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

    # Настройка устройства для вычислений
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Оптимизатор и планировщик обучения
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 5
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Функция потерь
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Обучение модели
    for epoch in range(epochs):
        print(f'Эпоха {epoch + 1}/{epochs}')
        train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler
        )
        val_loss, val_accuracy = eval_model(
            model,
            val_loader,
            loss_fn,
            device
        )
        print(f'Средний тренировочный loss: {train_loss}')
        print(f'Средний валидационный loss: {val_loss}')
        print(f'Точность на валидации: {val_accuracy}')
        print('-' * 50)

    # Сохранение модели и токенизатора
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Модель и токенизатор сохранены в директорию '{model_save_path}'.")

def check_files(data_path: str, required_files: list) -> None:
    """
    Проверяет наличие необходимых файлов в указанной директории.

    Args:
        data_path (str): Путь к директории с данными.
        required_files (list): Список имен необходимых файлов.

    Raises:
        FileNotFoundError: Если один из файлов не найден.
    """
    for file_name in required_files:
        if not os.path.isfile(os.path.join(data_path, file_name)):
            raise FileNotFoundError(f"Файл {file_name} не найден в директории {data_path}.")

def load_training_data(file_path: str) -> pd.DataFrame:
    """
    Загружает и возвращает тренировочные данные.

    Args:
        file_path (str): Путь к файлу с тренировочными данными.

    Returns:
        pd.DataFrame: Данные в виде DataFrame.
    """
    training_data = pd.read_csv(file_path, sep='\t', header=None)
    training_data.columns = ['id', 'Дата', 'Сумма', 'Назначение платежа', 'Категория']
    return training_data

def preprocess_text(text: str) -> str:
    """
    Предобрабатывает текст: удаляет лишние символы, приводит к нижнему регистру.

    Args:
        text (str): Исходный текст.

    Returns:
        str: Предобработанный текст.
    """
    if pd.isnull(text):
        return ''
    text = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class PaymentDataset(Dataset):
    """
    Кастомный датасет для платежей.

    Args:
        texts (pd.Series): Тексты для обучения.
        labels (pd.Series): Метки классов.
        tokenizer (BertTokenizerFast): Токенизатор BERT.
        max_len (int): Максимальная длина последовательности.
    """
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer: BertTokenizerFast, max_len: int = 128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, item: int) -> dict:
        encoding = self.tokenizer.encode_plus(
            self.texts[item],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def train_epoch(
    model: BertForSequenceClassification,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler
) -> float:
    """
    Выполняет одну эпоху обучения.

    Args:
        model (BertForSequenceClassification): Обучаемая модель.
        data_loader (DataLoader): Загрузчик обучающих данных.
        loss_fn (torch.nn.Module): Функция потерь.
        optimizer (torch.optim.Optimizer): Оптимизатор.
        device (torch.device): Устройство для вычислений.
        scheduler: Планировщик обучения.

    Returns:
        float: Среднее значение функции потерь за эпоху.
    """
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval_model(
    model: BertForSequenceClassification,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> tuple:
    """
    Оценивает модель на валидационной выборке.

    Args:
        model (BertForSequenceClassification): Обученная модель.
        data_loader (DataLoader): Загрузчик валидационных данных.
        loss_fn (torch.nn.Module): Функция потерь.
        device (torch.device): Устройство для вычислений.

    Returns:
        tuple: Среднее значение функции потерь и точность на валидации.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += torch.sum(preds == labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return avg_loss, accuracy.item()

if __name__ == '__main__':
    main()
