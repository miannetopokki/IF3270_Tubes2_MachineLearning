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
    '''
    input shape
    2D : (batch_size, input_length).
    output shape
    3D : (batch_size, input_length, output_dim).

    (1,4)
    Word index:
    "saya"  → 1
    "suka"  → 2
    "makan" → 3
    "nasi"  → 4


    (1,4,3)
    Embedding matrix:
    [
    [0.00, 0.00, 0.00],     # index 0 (biasanya untuk padding)
    [0.12, -0.05, 0.33],    # index 1 → "saya"
    [0.45, 0.10, -0.22],    # index 2 → "suka"
    [-0.13, 0.03, 0.15],    # index 3 → "makan"
    [0.08, -0.33, 0.41]     # index 4 → "nasi"
    ]
    '''
    def __init__(self, weights, config):
        super().__init__()
        self.embedding_matrix = weights[0].astype(np.float32)
        self.input_dim = config['input_dim']     # ukuran kosakata (jumlah token unik)
        self.output_dim = config['output_dim']   # dimensi vektor embedding tiap token

    def forward(self, inputs):
        inputs = inputs.astype(np.float32)
        # inputs shape: (batch_size, sequence_length)
        batch_size, seq_length = inputs.shape

        # output shape: (batch_size, sequence_length, embedding_dim)
        output = np.zeros((batch_size, seq_length, self.output_dim), dtype=np.float32)

        for i in range(batch_size):
            for j in range(seq_length):
                idx = int(inputs[i, j])  # token index
                if 0 <= idx < self.input_dim:
                    output[i, j] = self.embedding_matrix[idx]
        
        return output


class DenseLayer(BaseLayer):
  
    def __init__(self, weights, config):
        super().__init__()
        self.weights = weights[0].astype(np.float32)
        self.bias = weights[1].astype(np.float32)
        self.units = config['units']
        self.activation = config.get('activation', None)

    def forward(self, inputs):
        inputs = inputs.astype(np.float32)
        # y  = Wx + b
        outputs = np.dot(inputs, self.weights) + self.bias

        if self.activation == 'relu':
            outputs = np.maximum(0, outputs)
        elif self.activation == 'sigmoid':
            outputs = 1 / (1 + np.exp(-outputs))
        elif self.activation == 'softmax': 
            exp_outputs = np.exp(outputs - np.max(outputs, axis=-1, keepdims=True))
            outputs = exp_outputs / np.sum(exp_outputs, axis=-1, keepdims=True)
        elif self.activation == 'tanh':
            outputs = np.tanh(outputs)

        return outputs.astype(np.float32)


class DropoutLayer(BaseLayer):
    def __init__(self, weights, config):
        super().__init__()
        self.rate = config['rate']
        
    def forward(self, inputs):
        return (inputs.astype(np.float32)) * np.float32(1.0 - self.rate)

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
    
