import numpy as np
import sys
import os
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import NeuralNetworkModel, EmbeddingLayer, DenseLayer, DropoutLayer

np.set_printoptions(precision=10, suppress=True)
default_dtype = np.float32

def tanh(x):
    return np.tanh(x.astype(default_dtype)).astype(default_dtype)

class RNNCell:
    def __init__(self, weights, units):
        self.units = units
        self.W = weights[0].astype(default_dtype)
        self.U = weights[1].astype(default_dtype)
        self.b = weights[2].astype(default_dtype)

    def forward(self, x_t, h_prev):
        x_t = x_t.astype(default_dtype)
        h_prev = h_prev.astype(default_dtype)
        h_t = tanh(np.dot(x_t, self.W) + np.dot(h_prev, self.U) + self.b)
        return h_t.astype(default_dtype)

class RNNLayer:
    def __init__(self, weights, config):
        self.units = config['units']
        self.return_sequences = config.get('return_sequences', False)
        self.activation = config.get('activation', 'tanh')
        self.bidirectional = False
        self.rnn_cell = RNNCell(weights, self.units)

    def forward(self, inputs):
        inputs = inputs.astype(default_dtype)
        batch_size, timesteps, _ = inputs.shape
        h_t = np.zeros((batch_size, self.units), dtype=default_dtype)
        outputs = []

        for t in range(timesteps):
            x_t = inputs[:, t, :]
            h_t = self.rnn_cell.forward(x_t, h_t)
            outputs.append(h_t)

        outputs = np.stack(outputs, axis=1).astype(default_dtype)

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
        inputs = inputs.astype(default_dtype)
        batch_size, timesteps, _ = inputs.shape

        h_t_forward = np.zeros((batch_size, self.units), dtype=default_dtype)
        h_t_backward = np.zeros((batch_size, self.units), dtype=default_dtype)

        outputs_forward = []
        outputs_backward = []

        for t in range(timesteps):
            x_t_forward = inputs[:, t, :]
            h_t_forward = self.rnn_cell_forward.forward(x_t_forward, h_t_forward)
            outputs_forward.append(h_t_forward)

            x_t_backward = inputs[:, timesteps - 1 - t, :]
            h_t_backward = self.rnn_cell_backward.forward(x_t_backward, h_t_backward)
            outputs_backward.append(h_t_backward)

        outputs_forward = np.stack(outputs_forward, axis=1).astype(default_dtype)
        outputs_backward = np.stack(outputs_backward, axis=1).astype(default_dtype)

        output = np.concatenate((outputs_forward, outputs_backward), axis=-1).astype(default_dtype)

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
            config = self.layer_configs[i]
            weight = self.weights[i]

            if layer_type == 'Embedding':
                self.layers.append(EmbeddingLayer(weight, config))
            elif layer_type == 'SimpleRNN':
                self.layers.append(RNNLayer(weight, config))
            elif layer_type == 'Bidirectional' and 'SimpleRNN' in config['layer']['class_name']:
                self.layers.append(BidirectionalRNNLayer(weight, config['layer']['config']))
            elif layer_type == 'Dropout':
                self.layers.append(DropoutLayer(weight, config))
            elif layer_type == 'Dense':
                self.layers.append(DenseLayer(weight, config))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, inputs):
        x = inputs.astype(default_dtype)
        for layer in self.layers:
            x = layer.forward(x)
        return x

