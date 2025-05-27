import numpy as np
import sys
import os
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import NeuralNetworkModel, EmbeddingLayer, DenseLayer, DropoutLayer
class RNNCell:
    #Implementasi hitung hitung cell rnn disini
    def __init__(self):
        pass

    def forward(self):
        pass

class RNNLayer:
    def __init__(self, weights, config):
        #data dari config model keras, config bobot lstm sama rnn beda, yang rnn kurang tahu gw
        self.units = config['units']
        self.return_sequences = config.get('return_sequences', False)
        self.activation = config.get('activation', 'tanh')
        self.bidirectional = False
        self.rnn_cell = RNNCell()
        

    #implementasi perhitungan layer rnn disini, forward si cell rnn nya
    def forward(self):
        pass

class BidirectionalRNNLayer:
    #todo
    def __init__(self):
        pass
    def forward(self):
        pass

class RNNModel(NeuralNetworkModel):
    '''
    konsep:
    nerima input model keras ato path ke model keras (format {model_name}.h5)
    dibaca config di model, trus cek layer nya, append layer ke self.layer sesuai config
    pake kelas layer kita

    layer embedding,dropout,dense di parent class NeuralNetworkModel, karena lstm juga sama
    bedanya cuma diperhitungan cell layer aja, sama bidirectional
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
            elif layer_type == 'SimpleRNN':
                self.layers.append(RNNLayer(self.weights[i], self.layer_configs[i]))
            elif layer_type == 'Bidirectional' and 'SimpleRNN' in self.layer_configs[i]['layer']['class_name']:
                self.layers.append(BidirectionalRNNLayer())
            elif layer_type == 'Dropout':
                self.layers.append(DropoutLayer(self.weights[i], self.layer_configs[i]))
            elif layer_type == 'Dense':
                self.layers.append(DenseLayer(self.weights[i], self.layer_configs[i]))
    
    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
