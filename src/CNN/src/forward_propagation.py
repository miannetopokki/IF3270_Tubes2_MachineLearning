import numpy as np
import os
import tensorflow as tf 

class Conv2DLayer:
    def __init__(self, weights, biases, strides=(1,1), padding='same'):
 
        self.weights = weights
        self.biases = biases
        self.strides = strides
        self.padding = padding
        self.k_h, self.k_w, self.in_ch, self.num_filters = weights.shape
        self.s_y, self.s_x = strides

    def _pad_input(self, X_batch, pad_h, pad_w):

        if self.padding == 'valid':
            return X_batch, 0, 0 

        # P_h = ((H_out - 1) * S_h + K_h - H_in) / 2
        # P_w = ((W_out - 1) * S_w + K_w - W_in) / 2
        # Jika S=1, H_out = H_in, maka P_h = (K_h - 1) / 2
        # Pembulatan ke atas untuk padding_top/left dan ke bawah untuk padding_bottom/right

        pad_h_total = max((X_batch.shape[1] - 1) * self.s_y + self.k_h - X_batch.shape[1], 0) if self.padding == 'same' else 0
        pad_w_total = max((X_batch.shape[2] - 1) * self.s_x + self.k_w - X_batch.shape[2], 0) if self.padding == 'same' else 0

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        return np.pad(X_batch, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant'), pad_top, pad_left


    def forward(self, X_batch):

        batch_size, H_in, W_in, _ = X_batch.shape

        # Hitung dimensi output 
        if self.padding == 'same':
            H_out = int(np.ceil(float(H_in) / float(self.s_y)))
            W_out = int(np.ceil(float(W_in) / float(self.s_x)))
        elif self.padding == 'valid':
            H_out = int(np.ceil(float(H_in - self.k_h + 1) / float(self.s_y)))
            W_out = int(np.ceil(float(W_in - self.k_w + 1) / float(self.s_x)))
        else:
            raise ValueError("Padding tidak dikenal")

        X_padded, pad_h_before, pad_w_before = self._pad_input(X_batch, H_out, W_out) # pad_h/w_before untuk 'same'
        output = np.zeros((batch_size, H_out, W_out, self.num_filters))

        for i in range(batch_size): # Loop over batch
            img = X_padded[i]
            for h_out in range(H_out): # Loop over output height
                for w_out in range(W_out): # Loop over output width
                    # Tentukan irisan input
                    h_start = h_out * self.s_y
                    h_end = h_start + self.k_h
                    w_start = w_out * self.s_x
                    w_end = w_start + self.k_w

                    img_slice = img[h_start:h_end, w_start:w_end, :]

                    for f in range(self.num_filters): # Loop over filters
                        # Operasi konvolusi: element-wise multiplication dan sum
                        conv_val = np.sum(img_slice * self.weights[:, :, :, f]) + self.biases[f]
                        output[i, h_out, w_out, f] = conv_val
        return output

class ActivationLayer:
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def forward(self, X_batch):
        if self.activation_type == 'relu':
            return np.maximum(0, X_batch)
        elif self.activation_type == 'softmax':
            exp_scores = np.exp(X_batch - np.max(X_batch, axis=-1, keepdims=True)) 
            return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Aktivasi '{self.activation_type}' belum diimplementasikan.")

class PoolingLayer:
    def __init__(self, pool_size=(2,2), strides=(2,2), mode='max'):

        self.pool_h, self.pool_w = pool_size
        self.s_y, self.s_x = strides
        self.mode = mode

    def forward(self, X_batch):
        batch_size, H_in, W_in, C_in = X_batch.shape

        H_out = int((H_in - self.pool_h) / self.s_y) + 1
        W_out = int((W_in - self.pool_w) / self.s_x) + 1
        output = np.zeros((batch_size, H_out, W_out, C_in))

        for i in range(batch_size):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * self.s_y
                    h_end = h_start + self.pool_h
                    w_start = w_out * self.s_x
                    w_end = w_start + self.pool_w
                    input_slice = X_batch[i, h_start:h_end, w_start:w_end, :]

                    if self.mode == 'max':
                        output[i, h_out, w_out, :] = np.max(input_slice, axis=(0, 1))
                    elif self.mode == 'avg':
                        output[i, h_out, w_out, :] = np.mean(input_slice, axis=(0, 1))
        return output

class FlattenLayer:
    def forward(self, X_batch):
        return X_batch.reshape(X_batch.shape[0], -1)

class GlobalAveragePooling2DLayer:
    def forward(self, X_batch):
        # Input X_batch: (batch_size, height, width, channels)
        # Output: (batch_size, channels)
        return np.mean(X_batch, axis=(1, 2))


class DenseLayer:
    def __init__(self, weights, biases):

        self.weights = weights
        self.biases = biases

    def forward(self, X_batch):

        # Z = X . W + b
        return np.dot(X_batch, self.weights) + self.biases

class CustomCNN:
    def __init__(self, keras_model_path, model_config_name="model_for_scratch"):

        self.layers = []
        self._load_weights_and_build_layers(keras_model_path, model_config_name)

    def _load_weights_and_build_layers(self, keras_model_path, model_config_name):
   

        input_shape_cifar = (32, 32, 3)
        num_classes_cifar = 10

        
        from model_builder import build_cnn_model 


        #  Number of Convolution Layer
        if model_config_name == "cnn_1_conv_layers":
            conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
            ]
            pooling_type = 'max'
            global_pooling = 'flatten'
        elif model_config_name == "cnn_2_conv_layers":
            conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
                {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
            ]
            pooling_type = 'max'
            global_pooling = 'flatten'
        elif model_config_name == "cnn_3_conv_layers":
            conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
                {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
                {'filters': 128, 'kernel_size': (3,3), 'activation': 'relu'}
                  ]
            pooling_type = 'max'
            global_pooling = 'flatten'
        
        # Banyak Filter Per Layer Konvolusi
        elif model_config_name == "cnn_filters_16_32":
            conv_config = [
                {'filters': 16, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
            ]
            pooling_type = 'max'
            global_pooling = 'flatten'
        elif model_config_name == "cnn_filters_32_64":
            conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
                {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
            ]
            pooling_type = 'max'
            global_pooling = 'flatten'
        elif model_config_name == "cnn_filters_64_128":
            conv_config = [
                {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
                {'filters': 128, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
            ]
            pooling_type = 'max'
            global_pooling = 'flatten'
        
        elif model_config_name == "cnn_kernel_3x3_3x3":
            conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
                {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
            ]
            pooling_type = 'max'
            global_pooling= 'flatten'
        elif model_config_name == "cnn_kernel_mix_3x3_5x5":
            conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
                {'filters': 64, 'kernel_size': (5,5), 'activation': 'relu', 'padding': 'same'}
            ]
            pooling_type = 'max'
            global_pooling= 'flatten'
        elif model_config_name == "cnn_kernel_5x5_5x5":
            conv_config = [
                {'filters': 32, 'kernel_size': (5,5), 'activation': 'relu', 'padding': 'same'},
                {'filters': 64, 'kernel_size': (5,5), 'activation': 'relu', 'padding': 'same'}
            ]
            pooling_type = 'max'
            global_pooling= 'flatten'


        # Jenis Pooling Layer
        elif model_config_name == "cnn_max_pooling": 
             conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
                {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
            ]
             pooling_type = 'max'
             global_pooling = 'flatten'
        elif model_config_name == "cnn_avg_pooling":
             conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
                {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
            ]
             pooling_type = 'avg'
             global_pooling = 'flatten'
        else:
            raise ValueError(f"Konfigurasi untuk {model_config_name} tidak ditemukan.")


        temp_keras_model = build_cnn_model(input_shape_cifar, num_classes_cifar,
                                           conv_config, pooling_type, global_pooling)
        temp_keras_model.load_weights(keras_model_path)
        print(f"Bobot dari {keras_model_path} berhasil dimuat ke model Keras.")

        # 2. Ekstrak bobot dan bangun layer custom
        for keras_layer in temp_keras_model.layers:
            layer_name = keras_layer.name.lower() # Dapatkan nama layer
            # print(f"Processing Keras layer: {keras_layer.name} ({type(keras_layer)})")

            if isinstance(keras_layer, tf.keras.layers.Conv2D):
                weights, biases = keras_layer.get_weights()
                # Dapatkan stride dan padding dari konfigurasi layer Keras
                strides = keras_layer.strides
                padding_keras = keras_layer.padding
                self.layers.append(Conv2DLayer(weights, biases, strides=strides, padding=padding_keras))
                # print(f"  Added Conv2DLayer (from scratch). Padding: {padding_keras}, Strides: {strides}")
                if keras_layer.activation:
                    activation_name = tf.keras.activations.serialize(keras_layer.activation).lower()
                    self.layers.append(ActivationLayer(activation_name))
                    # print(f"  Added ActivationLayer (from scratch): {activation_name}")

            elif isinstance(keras_layer, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)):
                pool_size = keras_layer.pool_size
                strides = keras_layer.strides
                mode = 'max' if isinstance(keras_layer, tf.keras.layers.MaxPooling2D) else 'avg'
                self.layers.append(PoolingLayer(pool_size=pool_size, strides=strides, mode=mode))
                # print(f"  Added PoolingLayer (from scratch): {mode}, Pool size: {pool_size}, Strides: {strides}")

            elif isinstance(keras_layer, tf.keras.layers.Flatten):
                self.layers.append(FlattenLayer())
                # print(f"  Added FlattenLayer (from scratch).")

            elif isinstance(keras_layer, tf.keras.layers.GlobalAveragePooling2D):
                self.layers.append(GlobalAveragePooling2DLayer())
                # print(f"  Added GlobalAveragePooling2DLayer (from scratch).")

            elif isinstance(keras_layer, tf.keras.layers.Dense):
                weights, biases = keras_layer.get_weights()
                self.layers.append(DenseLayer(weights, biases))
                # print(f"  Added DenseLayer (from scratch).")
                if keras_layer.activation:
                    activation_name = tf.keras.activations.serialize(keras_layer.activation).lower()
                    if not (activation_name == 'softmax' and isinstance(self.layers[-2], DenseLayer) and keras_layer == temp_keras_model.layers[-1]):
                         self.layers.append(ActivationLayer(activation_name))
                        #  print(f"  Added ActivationLayer (from scratch): {activation_name}")
            elif isinstance(keras_layer, tf.keras.layers.InputLayer):
                print(f"  Skipping InputLayer.")
            else:
                print(f"  WARNING: Layer Keras {type(keras_layer)} tidak di-handle untuk forward pass manual.")
        print("Custom CNN model built with layers from Keras model.")


    def predict(self, X_batch):

        output = X_batch
        print("\n--- Starting Forward Propagation (from scratch) ---")
        for i, layer in enumerate(self.layers):
            output_before_layer = output # Untuk debugging
            output = layer.forward(output)
            # print(f"Layer {i+1} ({type(layer).__name__}): Input shape: {output_before_layer.shape}, Output shape: {output.shape}")
            if np.isnan(output).any():
                print(f"  WARNING: NaN detected in output of layer {i+1} ({type(layer).__name__})")
        print("--- Forward Propagation (from scratch) Finished ---")
        return output

# --- Fungsi untuk Pengujian ---
def test_forward_propagation(x_test_sample, y_test_sample_true_labels,
                             keras_model_weights_path,
                             keras_model_config_name, 
                             num_classes=10):

    from model_builder import build_cnn_model 

    if keras_model_config_name == "cnn_1_conv_layers":
            conv_config = [
                {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
            ]
            pooling = 'max'
            global_pool = 'flatten'
    elif keras_model_config_name == "cnn_2_conv_layers":
        conv_config = [
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
            {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
        ]
        pooling= 'max'
        global_pool = 'flatten'
    elif keras_model_config_name == "cnn_3_conv_layers":
        conv_config = [
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
            {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
            {'filters': 128, 'kernel_size': (3,3), 'activation': 'relu'}
            
        ]
        pooling= 'max'
        global_pool= 'flatten'    # Banyak Filter Per Layer Konvolusi
    elif keras_model_config_name == "cnn_filters_16_32":
        conv_config = [
            {'filters': 16, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
        ]
        pooling = 'max'
        global_pool = 'flatten'
    elif keras_model_config_name == "cnn_filters_32_64":
        conv_config = [
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
            {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
        ]
        pooling = 'max'
        global_pool = 'flatten'
    elif keras_model_config_name == "cnn_filters_64_128":
        conv_config = [
            {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'},
            {'filters': 128, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
        ]
        pooling = 'max'
        global_pool= 'flatten'#Ukuran kernel
    elif keras_model_config_name == "cnn_kernel_3x3_3x3":
        conv_config = [
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}, 
            {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}
        ]
        pooling = 'max'
        global_pool= 'flatten'
    elif keras_model_config_name == "cnn_kernel_mix_3x3_5x5":
        conv_config = [
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding': 'same'}, 
            {'filters': 64, 'kernel_size': (5,5), 'activation': 'relu', 'padding': 'same'}
        ]
        pooling = 'max'
        global_pool= 'flatten'
    elif keras_model_config_name == "cnn_kernel_5x5_5x5":
        conv_config = [
            {'filters': 32, 'kernel_size': (5,5), 'activation': 'relu', 'padding': 'same'}, 
            {'filters': 64, 'kernel_size': (5,5), 'activation': 'relu', 'padding': 'same'}
        ]
        pooling = 'max'
        global_pool= 'flatten'


    # Jenis Pooling Layer
    elif keras_model_config_name == "cnn_max_pooling": 
            conv_config = [
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
            {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
        ]
            pooling = 'max'
            global_pool = 'flatten'
    elif keras_model_config_name == "cnn_avg_pooling":
            conv_config = [
            {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'},
            {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'padding':'same'}
        ]
            pooling = 'avg'
            global_pool = 'flatten'
    else:
        raise ValueError(f"Konfigurasi Keras untuk '{keras_model_config_name}' tidak ditemukan untuk pengujian.")

    reference_keras_model = build_cnn_model((32,32,3), num_classes, conv_config, pooling, global_pool)
    reference_keras_model.load_weights(keras_model_weights_path)
    y_pred_keras_proba = reference_keras_model.predict(x_test_sample)
    y_pred_keras_labels = np.argmax(y_pred_keras_proba, axis=1)

    # 2. Dapatkan prediksi dari implementasi from scratch
    custom_model = CustomCNN(keras_model_weights_path, model_config_name=keras_model_config_name)
    y_pred_scratch_proba = custom_model.predict(x_test_sample)
    y_pred_scratch_labels = np.argmax(y_pred_scratch_proba, axis=1)

    # 3. Bandingkan hasil 
    print("\n--- Perbandingan Hasil Prediksi ---")
    print("Sample true labels:", y_test_sample_true_labels.flatten()[:10])
    print("Keras predicted labels:", y_pred_keras_labels[:10])
    print("Scratch predicted labels:", y_pred_scratch_labels[:10])

    # Bandingkan probabilitas output terakhir (sebelum argmax)
    print("\nProbabilitas output Keras (sampel pertama, 5 output pertama):", y_pred_keras_proba[0, :5])
    print("Probabilitas output Scratch (sampel pertama, 5 output pertama):", y_pred_scratch_proba[0, :5])

    
    from sklearn.metrics import f1_score
    f1_keras = f1_score(y_test_sample_true_labels, y_pred_keras_labels, average='macro')
    f1_scratch = f1_score(y_test_sample_true_labels, y_pred_scratch_labels, average='macro')

    print(f"\nMacro F1-Score (Keras): {f1_keras:.4f}")
    print(f"Macro F1-Score (Scratch): {f1_scratch:.4f}")

    if np.isclose(f1_keras, f1_scratch):
        print("Implementasi forward propagation from scratch KONSISTEN dengan Keras (berdasarkan F1-score).")
    else:
        print("Implementasi forward propagation from scratch TIDAK KONSISTEN dengan Keras (berdasarkan F1-score).")

if __name__ == '__main__':
    from data_loader import load_and_prepare_data
    (_, _), (_, _), (x_test, y_test) = load_and_prepare_data()

    num_test_samples = 10
    x_test_sample = x_test[:num_test_samples]
    y_test_sample_true_labels = y_test[:num_test_samples]


    MODEL_TO_TEST_SCRATCH = "cnn_2_conv_layers" 
    KERAS_WEIGHTS_PATH = f"saved_models/{MODEL_TO_TEST_SCRATCH}.weights.h5"

    if not os.path.exists(KERAS_WEIGHTS_PATH):
        print(f"ERROR: File bobot {KERAS_WEIGHTS_PATH} tidak ditemukan. Latih model terlebih dahulu.")
    else:
        print(f"\n--- Menguji Forward Propagation untuk Model: {MODEL_TO_TEST_SCRATCH} ---")
        test_forward_propagation(x_test_sample, y_test_sample_true_labels,
                                 KERAS_WEIGHTS_PATH,
                                 MODEL_TO_TEST_SCRATCH,
                                 num_classes=10)
