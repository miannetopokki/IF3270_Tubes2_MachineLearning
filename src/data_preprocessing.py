import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

class NusaXSentimentDataProcessor:

    def __init__(self, data_dir='indonesian', max_features=10000, sequence_length=100):
        self.data_dir = data_dir
        self.max_features = max_features
        self.sequence_length = sequence_length
        self.vectorize_layer = None
        self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
    def load_data(self):
        train_path = os.path.join(self.data_dir, 'train.csv')
        valid_path = os.path.join(self.data_dir, 'valid.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        try:
            train_data = pd.read_csv(train_path)
            val_data = pd.read_csv(valid_path)
            test_data = pd.read_csv(test_path)
            return train_data, val_data, test_data
            
        except Exception as e:
            pass
    
    def create_vectorize_layer(self, texts):
        self.vectorize_layer = keras.layers.TextVectorization(
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length
        )
        self.vectorize_layer.adapt(texts)
    
    def vectorize_text(self, text):
        return self.vectorize_layer(text)
    
    def prepare_data(self):
        train_data, val_data, test_data = self.load_data()
        unique_labels = set(train_data['label'].unique()) | set(val_data['label'].unique()) | set(test_data['label'].unique())
        print(f"Unique labels found: {unique_labels}")
        for label in unique_labels:
            if label not in self.label_mapping:
                print(f"Warning: Found unknown label '{label}'. Adding to label mapping.")
                self.label_mapping[label] = len(self.label_mapping)
        
        train_data['label_encoded'] = train_data['label'].map(self.label_mapping).astype(int)
        val_data['label_encoded'] = val_data['label'].map(self.label_mapping).astype(int)
        test_data['label_encoded'] = test_data['label'].map(self.label_mapping).astype(int)
        
        self.create_vectorize_layer(train_data['text'])
        
        x_train = np.array(self.vectorize_layer(train_data['text']))
        x_val = np.array(self.vectorize_layer(val_data['text']))
        x_test = np.array(self.vectorize_layer(test_data['text']))
        
        y_train = np.array(train_data['label_encoded'])
        y_val = np.array(val_data['label_encoded'])
        y_test = np.array(test_data['label_encoded'])
        
        print(f"Train data: {len(x_train)} samples")
        print(f"Validation data: {len(x_val)} samples")
        print(f"Test data: {len(x_test)} samples")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def get_vocabulary_size(self):
        if self.vectorize_layer is None:
            raise ValueError("Vectorize layer not created yet. Call prepare_data() first.")
        
        return len(self.vectorize_layer.get_vocabulary())
    
    def get_num_classes(self):
        return len(self.label_mapping)
    
