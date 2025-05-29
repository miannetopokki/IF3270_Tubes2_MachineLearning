import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod

class BaseLayer:
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, inputs):
        pass

class EmbeddingLayer(BaseLayer):
    #pakai embedding layer dari keras(dibolehin di spek)
    def __init__(self, weights, config):
        super().__init__()
        self.keras_embedding = tf.keras.layers.Embedding(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            weights=[weights[0]], 
            trainable=False
        )
        
    def forward(self, inputs):
        return self.keras_embedding(inputs).numpy()

class DenseLayer(BaseLayer):
    #pakai dense layer dari keras
    def __init__(self, weights, config):
        super().__init__()
        self.keras_dense = tf.keras.layers.Dense(
            units=config['units'],
            activation=config.get('activation', None),
            use_bias=True
        )
        input_dim = weights[0].shape[0]
        self.keras_dense.build((None, input_dim))
        self.keras_dense.set_weights([weights[0], weights[1]])
        self.keras_dense.trainable = False
        
    def forward(self, inputs):
        return self.keras_dense(inputs).numpy()

class DropoutLayer(BaseLayer):
    def __init__(self, weights, config):
        super().__init__()
        self.keras_dropout = tf.keras.layers.Dropout(
            rate=config['rate']
        )
        
    def forward(self, inputs):
        return self.keras_dropout(inputs, training=False).numpy()

class NeuralNetworkModel(ABC):
    def __init__(self, model_input=None):
        self.model = None
        self.weights = None
        self.layer_types = []
        self.layer_configs = []
        
        if model_input is not None:
            if isinstance(model_input, str):
                # input path file ke model.h5
                self.load_model(model_input)
            elif isinstance(model_input, tf.keras.Model):
                # input keras model
                self.model = model_input
                self._extract_model_info()
    
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        self._extract_model_info()
    

    #extract info dari config model keras
    def _extract_model_info(self):
        self.weights = []
        self.layer_types = []
        self.layer_configs = []
        
        for i, layer in enumerate(self.model.layers):
            self.layer_types.append(layer.__class__.__name__)
            self.layer_configs.append(layer.get_config())
            
            if layer.weights:
                layer_weights = []
                for w in layer.weights:
                    weight_array = w.numpy()
                    layer_weights.append(weight_array)
                self.weights.append(layer_weights)
            else:
                self.weights.append([])
    


    #info
    def print_info(self):
        print("\nModel Architecture Information:")
        print("==============================")
        
        for i, (layer_type, weights) in enumerate(zip(self.layer_types, self.weights)):
            print(f"\nLayer {i}: {layer_type}")
            print("------------------------")
                
            if layer_type == 'Embedding':
                if weights:
                    E = weights[0]  # Embedding matrix
                    print(f"E (Embedding Matrix):")
                    print(f"  Shape: {E.shape}")
                    print(f"  - rows: vocabulary size (|V|)")
                    print(f"  - cols: embedding dimension (d)")
                    
            elif layer_type == 'LSTM':
                if weights:
                    W = weights[0]  # Input weights (kernel)
                    U = weights[1]  # Recurrent weights (recurrent_kernel)
                    b = weights[2]  # Bias
                    
                    print("Weight Matrices:")
                    print(f"W (Input Weight Matrix):")
                    print(f"  Shape: {W.shape}")
                    print(f"  - rows: input dimension (d)")
                    print(f"  - cols: 4*h where h is hidden size (for i,f,g,o gates)")
                    
                    print(f"\nU (Recurrent Weight Matrix):")
                    print(f"  Shape: {U.shape}")
                    print(f"  - rows: hidden size (h)")
                    print(f"  - cols: 4*h (for i,f,g,o gates)")
                    
                    print(f"\nb (Bias Vector):")
                    print(f"  Shape: {b.shape}")
                    print(f"  - size: 4*h (bias for i,f,g,o gates)")
                    
                    h = U.shape[0]  # hidden size
                    print(f"\nLSTM Gates (each of size h={h}):")
                    print(f"  i: Input gate      (0:{h})")
                    print(f"  f: Forget gate     ({h}:{2*h})")
                    print(f"  g: Cell gate       ({2*h}:{3*h})")
                    print(f"  o: Output gate     ({3*h}:{4*h})")
                    
                    print("  c_t = f ⊙ c_{t-1} + i ⊙ g           # cell state")
                    print("  h_t = o ⊙ tanh(c_t)                  # hidden state")
                    
            elif layer_type == 'SimpleRNN':
                if weights:
                    W = weights[0]  # Input weights (kernel)
                    U = weights[1]  # Recurrent weights (recurrent_kernel)
                    b = weights[2]  # Bias
                    
                    print("Weight Matrices:")
                    print(f"W (Input Weight Matrix):")
                    print(f"  Shape: {W.shape}")
                    print(f"  - rows: input dimension (d)")
                    print(f"  - cols: hidden size (h)")
                    
                    print(f"\nU (Recurrent Weight Matrix):")
                    print(f"  Shape: {U.shape}")
                    print(f"  - rows: hidden size (h)")
                    print(f"  - cols: hidden size (h)")
                    
                    print(f"\nb (Bias Vector):")
                    print(f"  Shape: {b.shape}")
                    print(f"  - size: hidden size (h)")
                    
                    h = U.shape[0]  # hidden size
                    print(f"\nSimpleRNN Hidden Size: h={h}")
                    print("\nFormula:")
                    print("  h_t = tanh(W·x + U·h_{t-1} + b)     # hidden state")
                    
            elif layer_type == 'Dense':
                if weights:
                    W = weights[0]  # Weight matrix
                    b = weights[1]  # Bias
                    print(f"W (Weight Matrix):")
                    print(f"  Shape: {W.shape}")
                    print(f"  - rows: input features")
                    print(f"  - cols: output classes")
                    
                    print(f"\nb (Bias Vector):")
                    print(f"  Shape: {b.shape}")
            
            elif layer_type == 'Bidirectional':
                if weights:
                    forward_weights = weights[0:3]
                    backward_weights = weights[3:6]
                    print("Bidirectional RNN Layer:")
                    print("  Forward Layer:")
                    print(f"    W (Input Weight Matrix): {forward_weights[0].shape}")
                    print(f"    U (Recurrent Weight Matrix): {forward_weights[1].shape}")
                    print(f"    b (Bias Vector): {forward_weights[2].shape}")
                    print("  Backward Layer:")
                    print(f"    W (Input Weight Matrix): {backward_weights[0].shape}")
                    print(f"    U (Recurrent Weight Matrix): {backward_weights[1].shape}")
                    print(f"    b (Bias Vector): {backward_weights[2].shape}")
            
            if not weights:
                print("No weights in this layer")
            
            print(f"\nConfig:")
            print(self.layer_configs[i])
        
        print("\nFormulas:")
        print("=========")
        print("Embedding: x = E[input_indices]")
        print("SimpleRNN:")
        print("  h_t = tanh(W·x + U·h_{t-1} + b)     # hidden state")
        print("LSTM:")
        print("  i = σ(W_i·x + U_i·h_{t-1} + b_i)    # input gate")
        print("  f = σ(W_f·x + U_f·h_{t-1} + b_f)    # forget gate")
        print("  g = tanh(W_g·x + U_g·h_{t-1} + b_g) # cell gate")
        print("  o = σ(W_o·x + U_o·h_{t-1} + b_o)    # output gate")
        print("  c_t = f ⊙ c_{t-1} + i ⊙ g           # cell state")
        print("  h_t = o ⊙ tanh(c_t)                  # hidden state")
        print("Dense: y = W·x + b")

    def evaluate(self, x_test, y_test, average='macro', return_output=True):
        outputs = self.forward(x_test)
        predicted_classes = np.argmax(outputs, axis=1)
        
        f1 = f1_score(y_test, predicted_classes, average=average)
        print(f"F1 Score ({average}): {f1:.4f}")
        
        if return_output:
            return f1, predicted_classes, outputs
        
    
    @abstractmethod
    def forward(self, inputs):
        pass
    
    def predict(self, inputs):
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=1)
    
