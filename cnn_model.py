from  utils import np, load_csv
import matplotlib.pyplot as plt
from  tensorflow.keras import models, layers, losses, optimizers, activations, regularizers
from  sklearn.model_selection import train_test_split
from  tensorflow.keras.layers import Input, BatchNormalization, Conv1D, MaxPooling1D, Dropout
from  tensorflow.keras.layers import LeakyReLU

class CNN_Model:

    def __init__(self, num_samples, input_volume,filters = 5, kernel_size = 5, pool_size = 2 ,use_bias = False):
        
        self.model = models.Sequential()
        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='conv_1', use_bias=use_bias, input_shape=(num_samples, input_volume)))
        self.model.add(BatchNormalization(name='norm_1'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_1'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_1'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='conv_2', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_2'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_2'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_2'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='conv_3', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_3'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_3'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_3'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same',name='conv_4', use_bias=use_bias))
        self.model.add(BatchNormalization(name = 'norm_4'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_4'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_4'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='conv_5', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_5'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_5'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_5'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='conv_6', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_6'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_6'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_6'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same',name='conv_7', use_bias=use_bias))
        self.model.add(BatchNormalization(name = 'norm_7'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_7'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_7'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='conv_8', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_8'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_8'))
        self.model.add(MaxPooling1D(pool_size=pool_size, name='pool_8'))

        self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name='conv_9', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_9'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_9'))
        
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(units=200, activation=activations.linear, name='dense_5', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_dense_5'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_dense_5'))

        self.model.add(layers.Dense(units=50, activation=activations.linear, name='dense_6', use_bias=use_bias))
        self.model.add(BatchNormalization(name='norm_dense_6'))
        self.model.add(LeakyReLU(alpha=0.1, name='leaky_relu_dense_6'))

        self.model.add(layers.Dense(units=2, activation=activations.linear, name='dense_7', use_bias=use_bias))

    def get_model(self, learning_rate = 0.005, decay = 1e-4):
        self.model.compile(optimizer=optimizers.Adam(lr=learning_rate, decay=decay), loss=losses.mean_squared_error)
        return self.model
