import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf

def train_and_evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, epochs, batch_size, model_name="model", results_dir="results"):

    print(f"\n--- Melatih Model: {model_name} ---")

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=1)

    # Simpan bobot model
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    model.save_weights(f"saved_models/{model_name}_weights.weights.h5")
    print(f"Bobot model disimpan di: saved_models/{model_name}_weights.weights.h5")



    # Evaluasi pada test set
    print("\n--- Evaluasi pada Test Set ---")
    loss, accuracy, f1_macro_keras = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1-Score (Keras): {f1_macro_keras:.4f}") 


    # Plot training & validation loss
    plot_dir = os.path.join(results_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{model_name}_loss_plot.png"))
    plt.close()
    print(f"Grafik loss disimpan di: {os.path.join(plot_dir, f'{model_name}_loss_plot.png')}")

    return history, loss, accuracy, f1_macro_keras

if __name__ == '__main__':
    from data_loader import load_and_prepare_data
    from model_builder import build_cnn_model

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train)) 

    conv_config = [
        {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu'}
    ]
    base_model = build_cnn_model(input_shape, num_classes, conv_config, pooling_type='max')
    base_model.summary()

    history, test_loss, test_acc, test_f1 = train_and_evaluate_model(
        base_model, x_train, y_train, x_val, y_val, x_test, y_test,
        epochs=10,
        batch_size=64,
        model_name="base_model_test"
    )
    print(f"Hasil akhir evaluasi model dasar: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}, F1-Macro={test_f1:.4f}")