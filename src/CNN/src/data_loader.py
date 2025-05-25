import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_prepare_data():

    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisasi pixel values ke rentang [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Membagi training set menjadi training dan validation set (40k train, 10k val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"Jumlah data training: {x_train.shape[0]}")
    print(f"Jumlah data validasi: {x_val.shape[0]}")
    print(f"Jumlah data test: {x_test.shape[0]}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()
    print("Dimensi x_train:", x_train.shape)
    print("Dimensi y_train:", y_train.shape)
    print("Dimensi x_val:", x_val.shape)
    print("Dimensi y_val:", y_val.shape)
    print("Dimensi x_test:", x_test.shape)
    print("Dimensi y_test:", y_test.shape)
    print("Contoh label pertama y_train:", y_train[0])