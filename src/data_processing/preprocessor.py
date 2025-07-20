from typing import List
import pandas as pd
import re

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()

    def preprocess(self) -> pd.DataFrame:
        self.data['cleaned_text'] = self.data['text'].apply(self.clean_text)
        return self.data[['cleaned_text']]

    def extract_features(self) -> List[str]:
        return self.data['cleaned_text'].tolist()