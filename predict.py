import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


def main():
    """
    Основная функция для классификации новых данных с использованием предобученной модели.
    """
    # Путь к папке с данными
    data_path = 'data'
    model_path = os.path.join(data_path, 'model')
    input_file = os.path.join(data_path, 'payments_main.tsv')
    output_file = os.path.join(data_path, 'classified_payments.tsv')

    # Проверяем наличие необходимых файлов
    check_files(data_path, ['payments_main.tsv'])
    check_files(model_path, ['model.safetensors', 'config.json', 'vocab.txt', 'label_encoder.pkl'])

    # Загрузка данных
    new_data = load_new_data(input_file)

    # Предобработка текста
    new_data['text'] = new_data['Назначение платежа'].apply(preprocess_text)

    # Загрузка модели, токенизатора и LabelEncoder
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    # Настройка устройства для вычислений
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Классификация новых данных
    predictions = classify_new_data(model, new_data, tokenizer, device)

    # Преобразование предсказаний в категории
    new_data['Категория'] = label_encoder.inverse_transform(predictions)

    # Добавление номера строки
    new_data['Номер строки'] = new_data.index + 1

    # Сохранение результатов
    output = new_data[['Номер строки', 'Категория']]
    output.to_csv(output_file, sep='\t', index=False, header=None)

    print(f"Классификация завершена. Результаты сохранены в файле '{output_file}'.")

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

def load_new_data(file_path: str) -> pd.DataFrame:
    """
    Загружает и возвращает новые данные для классификации.

    Args:
        file_path (str): Путь к файлу с новыми данными.

    Returns:
        pd.DataFrame: Данные в виде DataFrame.
    """
    new_data = pd.read_csv(file_path, sep='\t', header=None)
    new_data.columns = ['id', 'Дата', 'Сумма', 'Назначение платежа']
    return new_data

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

def classify_new_data(
    model: BertForSequenceClassification,
    new_data: pd.DataFrame,
    tokenizer: BertTokenizerFast,
    device: torch.device
) -> list:
    """
    Классифицирует новые данные с использованием обученной модели.

    Args:
        model (BertForSequenceClassification): Обученная модель.
        new_data (pd.DataFrame): Новые данные для классификации.
        tokenizer (BertTokenizerFast): Токенизатор BERT.
        device (torch.device): Устройство для вычислений.

    Returns:
        list: Список предсказанных меток классов.
    """
    # Создание датасета
    new_dataset = NewPaymentDataset(new_data['text'], tokenizer)
    new_loader = DataLoader(new_dataset, batch_size=16)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in new_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())

    return predictions

class NewPaymentDataset(Dataset):
    """
    Кастомный датасет для новых платежей без меток.

    Args:
        texts (pd.Series): Тексты для классификации.
        tokenizer (BertTokenizerFast): Токенизатор BERT.
        max_len (int): Максимальная длина последовательности.
    """
    def __init__(self, texts: pd.Series, tokenizer: BertTokenizerFast, max_len: int = 128):
        self.texts = texts.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

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
            'attention_mask': encoding['attention_mask'].flatten()
        }

if __name__ == '__main__':
    main()
