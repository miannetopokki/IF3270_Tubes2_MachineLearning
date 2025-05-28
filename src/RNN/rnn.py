import numpy as np
import sys
import os
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import NeuralNetworkModel, EmbeddingLayer, DenseLayer, DropoutLayer

def tanh(x):
    return np.tanh(x)

class RNNCell:
    def __init__(self, weights, units):
        
        self.units = units
        self.W = weights[0] 
        self.U = weights[1] 
        self.b = weights[2] 
        pass

    def forward(self, x_t, h_prev):
        #NOTE: Apa perlu implemene activation function lain selain tanh?
        h_t = tanh(np.dot(x_t, self.W) + np.dot(h_prev, self.U) + self.b)
        
        return h_t

class RNNLayer:
    def __init__(self, weights, config):
        
        self.units = config['units']
        self.return_sequences = config.get('return_sequences', False)
        self.activation = config.get('activation', 'tanh')
        self.bidirectional = False
        self.rnn_cell = RNNCell(weights, self.units)
        

    def forward(self, inputs):
        
        batch_size, timesteps, _ = inputs.shape
        h_t = np.zeros((batch_size, self.units))  
        
        outputs = []
        
        for t in range(timesteps):
            x_t = inputs[:, t, :]
            h_t = self.rnn_cell.forward(x_t, h_t)
            outputs.append(h_t)
            
        outputs = np.stack(outputs, axis=1) 
        
        if self.return_sequences:
            return outputs
        else:
            return outputs[:, -1, :]

class BidirectionalRNNLayer:
    def __init__(self, weights, config):
        
        self.units = config['units']
        self.return_sequences = config.get('return_sequences', False)
        self.activation = config.get('activation', 'tanh')
        self.bidirectional = True
        self.rnn_cell_forward = RNNCell(weights[:3], self.units)
        self.rnn_cell_backward = RNNCell(weights[3:], self.units)
        
    def forward(self, inputs):
        
        batch_size, timesteps, _ = inputs.shape
        h_t_forward = np.zeros((batch_size, self.units))
        h_t_backward = np.zeros((batch_size, self.units))
        
        outputs_forward = []
        outputs_backward = []
        
        for t in range(timesteps):
            x_t_forward = inputs[:, t, :]
            h_t_forward = self.rnn_cell_forward.forward(x_t_forward, h_t_forward)
            outputs_forward.append(h_t_forward)
            
            x_t_backward = inputs[:, timesteps - 1 - t, :]
            h_t_backward = self.rnn_cell_backward.forward(x_t_backward, h_t_backward)
            outputs_backward.append(h_t_backward)
            
        outputs_forward = np.stack(outputs_forward, axis=1)
        outputs_backward = np.stack(outputs_backward[::-1], axis=1) 
        
        # NOTE: apa perlu implement metode lain untuk menggabungkan output forward dan backward?
        output = np.concatenate((outputs_forward, outputs_backward), axis=-1) 
        
        if self.return_sequences:
            return output
        else:
            return output[:, -1, :]
        

class RNNModel(NeuralNetworkModel):
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
            elif layer_type == 'SimpleRNN':
                self.layers.append(RNNLayer(self.weights[i], self.layer_configs[i]))
            elif layer_type == 'Bidirectional' and 'SimpleRNN' in self.layer_configs[i]['layer']['class_name']:
                self.layers.append(BidirectionalRNNLayer(self.weights[i], self.layer_configs[i]['layer']['config']))
            elif layer_type == 'Dropout':
                self.layers.append(DropoutLayer(self.weights[i], self.layer_configs[i]))
            elif layer_type == 'Dense':
                self.layers.append(DenseLayer(self.weights[i], self.layer_configs[i]))
    
    def forward(self, inputs):
        
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
