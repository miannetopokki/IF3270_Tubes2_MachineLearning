import numpy as np
import sys
import os
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import NeuralNetworkModel, EmbeddingLayer, DenseLayer, DropoutLayer
default_dtype = np.float32

def sigmoid(x):
    return (1 / (1 + np.exp(-x.astype(default_dtype)))).astype(default_dtype)

def tanh(x):
    return np.tanh(x.astype(default_dtype)).astype(default_dtype)

'''

kernel: W → bobot dari input x_t ke gate, shape (input_dim, 4 * units)

recurrent_kernel: U → bobot dari hidden state h_{t-1} ke gate, shape (units, 4 * units)

bias: b → vektor bias per gate, shape (4 * units,)

4 * unit karena ada 4 gate
Weight Matrices:
W (Input Weight Matrix):
  Shape: (1, 40)
  - rows: input dimension (d)
  - cols: 4*h where h is hidden size (for i,f,g,o gates)

U (Recurrent Weight Matrix):
  Shape: (10, 40)
  - rows: hidden size (h)
...
  o = σ(W_o·x + U_o·h_{t-1} + b_o)    # output gate
  c_t = f ⊙ c_{t-1} + i ⊙ g           # cell state
  h_t = o ⊙ tanh(c_t)                  # hidden state
Dense: y = W·x + b
'''
class LSTMCell:
    def __init__(self, weights, units):
        # weights: [kernel/W, recurrent_kernel/U, bias]
        self.units = units
        self.W = weights[0].astype(default_dtype)         # shape (input_dim, 4*units)
        self.U = weights[1].astype(default_dtype)         # shape (units, 4*units)
        self.b = weights[2].astype(default_dtype)         # shape (4*units,)
        
    def forward(self, x_t, h_prev, c_prev):
        x_t = x_t.astype(default_dtype)
        h_prev = h_prev.astype(default_dtype)
        c_prev = c_prev.astype(default_dtype)
        
        #W * x + U * out{t-1} + b
        z = np.dot(x_t, self.W) + np.dot(h_prev, self.U) + self.b
        z = z.astype(default_dtype)
        
        f_t = sigmoid(z[:, 0*self.units:1*self.units])   # forget gate
        i_t = sigmoid(z[:, 1*self.units:2*self.units])   # input gate
        c_hat_t = tanh(z[:, 2*self.units:3*self.units])  # candidate cell
        o_t = sigmoid(z[:, 3*self.units:4*self.units])   # output gate
        
        c_t = f_t * c_prev + i_t * c_hat_t 
        h_t = o_t * tanh(c_t)
        
        return h_t, c_t



class LSTMLayer:
    def __init__(self, weights, config):
        self.units = config['units']
        self.return_sequences = config.get('return_sequences', False)
        self.activation = config.get('activation', 'tanh')
        self.recurrent_activation = config.get('recurrent_activation', 'sigmoid')
        self.bidirectional = False
        self.lstm_cell = LSTMCell(weights, self.units)
        
    def forward(self, inputs):
        # inputs shape: (batch_size, timesteps, input_dim)
        inputs = inputs.astype(default_dtype)
        batch_size, timesteps, _ = inputs.shape
        
        #init awal, h_t dan c_t 0
        h_t = np.zeros((batch_size, self.units), dtype=default_dtype)
        c_t = np.zeros((batch_size, self.units), dtype=default_dtype)
        
        outputs = []
        for t in range(timesteps):
            x_t = inputs[:, t, :]  # input pada timestep t
            h_t, c_t = self.lstm_cell.forward(x_t, h_t, c_t)
            outputs.append(h_t)
        
        outputs = np.stack(outputs, axis=1).astype(default_dtype)  # shape (batch_size, timesteps, units)
        
        if self.return_sequences:
            return outputs
        else: 
            return outputs[:, -1, :]  # hanya output terakhir
        
class BidirectionalLSTMLayer:
    def __init__(self, weights, config):
        self.units = config['units']
        self.return_sequences = config.get('return_sequences', False)
        self.activation = config.get('activation', 'tanh')
        self.recurrent_activation = config.get('recurrent_activation', 'sigmoid')
        self.bidirectional = True
        
        self.lstm_cell_forward = LSTMCell(weights[:3], self.units)
        self.lstm_cell_backward = LSTMCell(weights[3:], self.units)

    def forward(self, inputs):
        inputs = inputs.astype(default_dtype)
        batch_size, timesteps, _ = inputs.shape
        
        
        h_t_forward = np.zeros((batch_size, self.units), dtype=default_dtype)
        c_t_forward = np.zeros((batch_size, self.units), dtype=default_dtype)
        h_t_backward = np.zeros((batch_size, self.units), dtype=default_dtype)
        c_t_backward = np.zeros((batch_size, self.units), dtype=default_dtype)
        
        outputs_forward = []
        outputs_backward = []
        
        for t in range(timesteps):
            x_t_forward = inputs[:, t, :]
            h_t_forward, c_t_forward = self.lstm_cell_forward.forward(x_t_forward, h_t_forward, c_t_forward)
            outputs_forward.append(h_t_forward)
        
        for t in range(timesteps):
            x_t_backward = inputs[:, timesteps - 1 - t, :]
            h_t_backward, c_t_backward = self.lstm_cell_backward.forward(x_t_backward, h_t_backward, c_t_backward)
            outputs_backward.append(h_t_backward)
        
       
        outputs_forward = np.stack(outputs_forward, axis=1).astype(default_dtype)
        outputs_backward = np.stack(outputs_backward, axis=1).astype(default_dtype)
        
        output = np.concatenate((outputs_forward, outputs_backward), axis=-1).astype(default_dtype)
        
        if self.return_sequences:
            return output
        else:
            return output[:, -1, :]

class LSTMModel(NeuralNetworkModel):
    '''
    konsep:
    nerima input model ato path ke model (format {model_name}.h5)
    dibaca config di model, trus cek layer nya, append layer ke self.layer sesuai config
    pake kelas layer kita
    '''
    def __init__(self, model_input=None):
        super().__init__(model_input)
        self.layers = []
        if model_input is not None:
            self._build_layers()
    
    def _build_layers(self):
        self.layers = []
        
        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'Embedding':
                self.layers.append(EmbeddingLayer(self.weights[i], self.layer_configs[i]))
            elif layer_type == 'LSTM':
                self.layers.append(LSTMLayer(self.weights[i], self.layer_configs[i]))
            elif layer_type == 'Bidirectional' and 'LSTM' in self.layer_configs[i]['layer']['class_name']:
                self.layers.append(BidirectionalLSTMLayer(self.weights[i], self.layer_configs[i]['layer']['config']))
            elif layer_type == 'Dropout':
                self.layers.append(DropoutLayer(self.weights[i], self.layer_configs[i]))
            elif layer_type == 'Dense':
                self.layers.append(DenseLayer(self.weights[i], self.layer_configs[i]))
    
    def forward(self, inputs):
        x = inputs.astype(default_dtype)
        for layer in self.layers:
            x = layer.forward(x)
        return x
