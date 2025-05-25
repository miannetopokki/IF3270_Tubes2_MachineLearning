import tensorflow as tf
from data_loader import load_and_prepare_data
from model_builder import build_cnn_model
from train_evaluate import train_and_evaluate_model
import numpy as np
import os

# Konfigurasi Umum
EPOCHS = 10
BATCH_SIZE = 64
NUM_CLASSES = 10 
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
SCORES_FILE = os.path.join(RESULTS_DIR, "scores.txt")

def log_results(file_path, experiment_name, f1_score_val, loss_val, acc_val, details=""):
    with open(file_path, "a") as f:
        f.write(f"Eksperimen: {experiment_name}\n")
        if details:
            f.write(f"  Details: {details}\n")
        f.write(f"  Macro F1-Score: {f1_score_val:.4f}\n")
        f.write(f"  Test Loss: {loss_val:.4f}\n")
        f.write(f"  Test Accuracy: {acc_val:.4f}\n")
        f.write("-" * 30 + "\n")

def run_experiments():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()
    input_shape = x_train.shape[1:]

    all_results = {} # Untuk menyimpan hasil F1-score dari setiap eksperimen

    # Hapus file skor lama jika ada
    if os.path.exists(SCORES_FILE):
        os.remove(SCORES_FILE)

    # --- 1. Pengaruh Jumlah Layer Konvolusi ---
    print("\n=== Eksperimen: Pengaruh Jumlah Layer Konvolusi ===")
    num_conv_layers_variations = [
        ("1_conv_layers", [{'filters': 32, 'kernel_size': (3,3), 'activation': 'relu'}]),
        ("2_conv_layers", [{'filters': 32, 'kernel_size': (3,3), 'activation': 'relu'},
                           {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu'}]),
        ("3_conv_layers", [{'filters': 32, 'kernel_size': (3,3), 'activation': 'relu'},
                           {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu'},
                           {'filters': 128, 'kernel_size': (3,3), 'activation': 'relu'}])
    ]

    for name, config in num_conv_layers_variations:
        print(f"\n--- Variasi: {name} ---")
        model_name = f"cnn_{name}"
        model = build_cnn_model(input_shape, NUM_CLASSES, config, pooling_type='max')
        model.summary() # Tampilkan summary untuk verifikasi
        _, loss, acc, f1 = train_and_evaluate_model(
            model, x_train, y_train, x_val, y_val, x_test, y_test,
            EPOCHS, BATCH_SIZE, model_name, RESULTS_DIR
        )
        all_results[model_name] = f1
        log_results(SCORES_FILE, model_name, f1, loss, acc, f"Config: {config}")
        tf.keras.backend.clear_session() # Reset state Keras

    # --- 2. Pengaruh Banyak Filter per Layer Konvolusi ---
    print("\n=== Eksperimen: Pengaruh Banyak Filter per Layer Konvolusi ===")
    #2 layer konvolusi sebagai dasar
    num_filters_variations = [
        ("filters_16_32", [{'filters': 16, 'kernel_size': (3,3)}, {'filters': 32, 'kernel_size': (3,3)}]),
        ("filters_32_64", [{'filters': 32, 'kernel_size': (3,3)}, {'filters': 64, 'kernel_size': (3,3)}]), # Default
        ("filters_64_128", [{'filters': 64, 'kernel_size': (3,3)}, {'filters': 128, 'kernel_size': (3,3)}])
    ]
    base_conv_layers_for_filters = 2 

    for name, config in num_filters_variations:
        print(f"\n--- Variasi: {name} ---")
        model_name = f"cnn_{name}"

        model = build_cnn_model(input_shape, NUM_CLASSES, config, pooling_type='max')
        model.summary()
        _, loss, acc, f1 = train_and_evaluate_model(
            model, x_train, y_train, x_val, y_val, x_test, y_test,
            EPOCHS, BATCH_SIZE, model_name, RESULTS_DIR
        )
        all_results[model_name] = f1
        log_results(SCORES_FILE, model_name, f1, loss, acc, f"Config: {config}")
        tf.keras.backend.clear_session()

    # --- 3. Pengaruh Ukuran Filter per Layer Konvolusi ---
    print("\n=== Eksperimen: Pengaruh Ukuran Filter per Layer Konvolusi ===")
    # 2 layer konvolusi dengan filter [32, 64] sebagai dasar
    kernel_size_variations = [
        ("kernel_3x3_3x3", [{'filters': 32, 'kernel_size': (3,3)}, {'filters': 64, 'kernel_size': (3,3)}]), # Default
        ("kernel_5x5_5x5", [{'filters': 32, 'kernel_size': (5,5)}, {'filters': 64, 'kernel_size': (5,5)}]),
        ("kernel_mix_3x3_5x5", [{'filters': 32, 'kernel_size': (3,3)}, {'filters': 64, 'kernel_size': (5,5)}])
    ]

    for name, config in kernel_size_variations:
        print(f"\n--- Variasi: {name} ---")
        model_name = f"cnn_{name}"
        model = build_cnn_model(input_shape, NUM_CLASSES, config, pooling_type='max')
        model.summary()
        _, loss, acc, f1 = train_and_evaluate_model(
            model, x_train, y_train, x_val, y_val, x_test, y_test,
            EPOCHS, BATCH_SIZE, model_name, RESULTS_DIR
        )
        all_results[model_name] = f1
        log_results(SCORES_FILE, model_name, f1, loss, acc, f"Config: {config}")
        tf.keras.backend.clear_session()

    # --- 4. Pengaruh Jenis Pooling Layer ---
    print("\n=== Eksperimen: Pengaruh Jenis Pooling Layer ===")
    # 2 layer [32,64] kernel (3,3)
    base_config_for_pooling = [
        {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu'}
    ]
    pooling_types = [
        ("max_pooling", "max"),
        ("avg_pooling", "avg")
    ]

    for name, pool_type in pooling_types:
        print(f"\n--- Variasi: {name} ---")
        model_name = f"cnn_{name}"
        model = build_cnn_model(input_shape, NUM_CLASSES, base_config_for_pooling, pooling_type=pool_type)
        model.summary()
        _, loss, acc, f1 = train_and_evaluate_model(
            model, x_train, y_train, x_val, y_val, x_test, y_test,
            EPOCHS, BATCH_SIZE, model_name, RESULTS_DIR
        )
        all_results[model_name] = f1
        log_results(SCORES_FILE, model_name, f1, loss, acc, f"Pooling type: {pool_type}")
        tf.keras.backend.clear_session()

    print("\n=== Ringkasan F1-Score Semua Eksperimen ===")
    for model_name, f1_val in all_results.items():
        print(f"{model_name}: {f1_val:.4f}")

if __name__ == '__main__':
    run_experiments()