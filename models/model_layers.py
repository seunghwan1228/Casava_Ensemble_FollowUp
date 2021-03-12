import tensorflow as tf


class ResnetBlock(tf.keras.layers.LayeR):
    """
    Residual Network
    https://arxiv.org/pdf/1512.03385.pdf

    Apply Bottleneck block for ResNet-50/101/152

    The Author applied Batchnormalization before activation - (3.4)
    """
    def __init__(self, conv_1_2_filters, conv_3_filters, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.conv_1_2_filters = conv_1_2_filters
        self.conv_3_filters = conv_3_filters
        
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.conv_1_2_filters,
                                             kernel_size=(1, 1),
                                             strides=(1, 1),
                                             padding='same')
        self.conv_1_bn = tf.keras.layers.BatchNormalization()
        self.conv_1_act = tf.keras.layers.Activation('relu')
        
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.conv_1_2_filters,
                                            kernel_size=(3,3),
                                            strides=(1,1),
                                            padding='same')
        self.conv_2_bn = tf.keras.layers.BatchNormalization()
        self.conv_2_act = tf.keras.layers.Activation('relu')
        
        self.conv_3 = tf.keras.layers.Conv2D(self.conv_3_filters, 
                                            kernel_size=(1,1),
                                            strides=(1,1),
                                            padding='same')
        self.conv_3_bn = tf.keras.layers.BatchNormalization()
        self.conv_3_act = tf.keras.layers.Activation('relu')
        
    def call(self, inputs):
        res = inputs
        x = self.conv_1(inputs)
        x = self.conv_1_bn(x)
        x = self.conv_1_act(x)
        x = self.conv_2(x)
        x = self.conv_2_bn(x)
        x = self.conv_2_act(x)
        x = self.conv_3(x)
        x = self.conv_3_bn(x)
        x = self.conv_3_act(x)
        
        return res + x



class ResNextBlock(tf.keras.layers.Layer):
    """
    Type - A
    Building Blocks of ResNeXt 
    (a) Aggregated residual transformations
    """
    def __init__(self, conv_1_2_filters, conv_3_filters, cardinality, **kwargs):
        self.conv_1_2_filters = conv_1_2_filters
        self.conv_3_filters = conv_3_filters
        self.cardinality = cardinality
        
        self.resnext_block = [ResnetBlock(conv_1_2_filters=self.conv_1_2_filters,
                                          conv_3_filters=self.conv_3_filters) for _ in range(self.cardinality)]

        self.addiction = tf.keras.layers.Add()
        
    def call(self, inputs):
        res = inputs        
        block_results = []
        for cardi_layer in self.resnext_block:
            block_result = cardi_layer(inputs)
            block_results.append(block_result)
        
        x = self.addiction(block_results)
        
        return res + x