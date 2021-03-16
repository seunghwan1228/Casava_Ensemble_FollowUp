import tensorflow as tf
from models.model_layers import ResnextBlock, ResNextSumBlock, ResNextConcatBlock

class ResNextConcat50(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(ResNextConcat50, self).__init__(**kwargs)
        self.config = config
        
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.config['conv1_filters'],
                                             kernel_size=(self.config['conv1_kernel'], self.config['conv1_kernel']),
                                             strides=(self.config['conv1_strides'], self.config['conv1_strides']),
                                             padding='same',
                                             activation='relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(self.config['maxpool_size'], self.config['maxpool_size']),
                                                  strides=(self.config['maxpool_stride'], self.config['maxpool_stride']))
        
        self.conv_2_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv2_1_2_filters'],
                                                 conv_3_filters=self.config['conv2_3_filters'], 
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv2_layers'])]

        self.conv_3_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv3_1_2_filters'],
                                                 conv_3_filters=self.config['conv3_3_filters'],
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv3_layers'])]

        self.conv_4_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv4_1_2_filters'],
                                                 conv_3_filters=self.config['conv4_3_filters'],
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv4_layers'])]
    
        self.conv_5_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv5_1_2_filters'],
                                                 conv_3_filters=self.config['conv5_3_filters'],
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv5_layers'])]
    
        self.ga_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(config.num_classes)
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pool(x)    
        for conv2_layer in self.conv_2_blocks:
            x = conv2_layer(x)
        for conv3_layer in self.conv_3_blocks:
            x = conv3_layer(x)
        for conv4_layer in self.conv_4_blocks:
            x = conv4_layer(x)
        for conv5_layer in self.conv_5_blocks:
            x = conv5_layer
        x = self.ga_pool(x)
        return self.output_layer(x)


class ResNextSum50(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(ResNextSum50, self).__init__(**kwargs)
        self.config = config
        
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.config['conv1_filters'],
                                             kernel_size=(self.config['conv1_kernel'], self.config['conv1_kernel']),
                                             strides=(self.config['conv1_strides'], self.config['conv1_strides']),
                                             padding='same',
                                             activation='relu')
        
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(self.config['maxpool_size'], self.config['maxpool_size']),
                                                  strides=(self.config['maxpool_stride'], self.config['maxpool_stride']))
        
        self.conv_2_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv2_1_2_filters'],
                                              conv_3_filters=self.config['conv2_3_filters'], 
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv2_layers'])]

        self.conv_3_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv3_1_2_filters'],
                                              conv_3_filters=self.config['conv3_3_filters'],
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv3_layers'])]

        self.conv_4_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv4_1_2_filters'],
                                              conv_3_filters=self.config['conv4_3_filters'],
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv4_layers'])]
    
        self.conv_5_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv5_1_2_filters'],
                                              conv_3_filters=self.config['conv5_3_filters'],
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv5_layers'])]
    
        self.ga_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(config.num_classes)
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pool(x)    
        for conv2_layer in self.conv_2_blocks:
            x = conv2_layer(x)
        for conv3_layer in self.conv_3_blocks:
            x = conv3_layer(x)
        for conv4_layer in self.conv_4_blocks:
            x = conv4_layer(x)
        for conv5_layer in self.conv_5_blocks:
            x = conv5_layer
        x = self.ga_pool(x)
        return self.output_layer(x)