import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes, conv_layers_config, pooling_type='max', global_pooling_type='flatten'):

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Layer Konvolusi dan Pooling
    for config in conv_layers_config:
        model.add(layers.Conv2D(filters=config['filters'],
                                kernel_size=config['kernel_size'],
                                activation=config.get('activation', 'relu'), 
                                padding=config.get('padding', 'same'))) 
        if pooling_type == 'max':
            model.add(layers.MaxPooling2D((2, 2)))
        elif pooling_type == 'avg':
            model.add(layers.AveragePooling2D((2, 2)))


    # Layer Flatten atau Global Pooling
    if global_pooling_type == 'flatten':
        model.add(layers.Flatten())
    elif global_pooling_type == 'global_avg':
        model.add(layers.GlobalAveragePooling2D())
    else:
        raise ValueError("global_pooling_type tidak valid")

    # Dense Layer
    model.add(layers.Dense(128, activation='relu')) 
    model.add(layers.Dense(num_classes, activation='softmax')) 

    # Kompilasi Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score_macro')]) 

    return model

if __name__ == '__main__':
    dummy_input_shape = (32, 32, 3)
    dummy_num_classes = 10
    dummy_conv_config = [
        {'filters': 32, 'kernel_size': (3,3)},
        {'filters': 64, 'kernel_size': (3,3)}
    ]
    model = build_cnn_model(dummy_input_shape, dummy_num_classes, dummy_conv_config, pooling_type='max')
    model.summary()