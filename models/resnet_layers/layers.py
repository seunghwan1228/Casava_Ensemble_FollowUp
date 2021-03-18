import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import identity


class ConvOps(tf.keras.layers.Layer):
    """[summary]
        Operating Convolution 
        Conv - Norn - Activation
    """
    def __init__(self, filters, kernel_size, strides, padding, groups, **kwargs):
        super(ConvOps, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups)
        self.norm = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        return self.act(x)


class ResNetBlock(tf.keras.layers.Layer):
    """[summary]
        Operating Residual Connection Block
        
        Conv1 - Norm - Activation - Conv2 - Norm - Activation 
        Input                                                  - Connection(+ Operation)
        --------------------------------------------------------------------------------
        
        Pair as "Deep Residual Learning for Image Recognition" Fig.3 < Straight Line >
    """
    def __init__(self, conv_filters, kernel_size, strides_1, strides_2, padding, groups, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv_1 = ConvOps(filters=conv_filters, kernel_size=kernel_size, strides=strides_1, padding=padding, groups=groups)
        self.conv_2 = ConvOps(filters=conv_filters, kernel_size=kernel_size, strides=strides_2, padding=padding, groups=groups)
    
    def call(self, inputs):
        residual = inputs
        conv1_output = self.conv_1(inputs)
        conv2_output = self.conv_2(conv1_output)
        return conv2_output + residual
    
class ResNetBlockExpand(tf.keras.layers.Layer):
    """[summary]
        Operating Residual Connection Block
        
        Conv1 - Norm - Activation - Conv2 - Norm - Activation 
        Input - Conv(1x1)                                      - Connection(+ Operation)
        -------------------------------------------------------------------------------- 
        
        Pair as "Deep Residual Learning for Image Recognition" Fig.3 < Dotted Line >
    """
    def __init__(self, conv_filters, kernel_size, strides_1, strides_2, padding, groups, **kwargs):
        super(ResNetBlockExpand, self).__init__(**kwargs)
        self.conv_1 = ConvOps(filters=conv_filters, kernel_size=kernel_size, strides=strides_1, padding=padding, groups=groups)
        self.conv_2 = ConvOps(filters=conv_filters, kernel_size=kernel_size, strides=strides_2, padding=padding, groups=groups)
        self.conv_expand = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=(1,1), strides=strides_1, padding='same')
        
    def call(self, inputs):
        residual = inputs
        
        conv1_output = self.conv_1(inputs)
        conv2_output = self.conv_2(conv1_output)
        
        conv_expanded = self.conv_expand(residual)
        return conv_expanded + conv2_output
    

        
    

if __name__ == "__main__":
    
    def experiment(input, layer):
        result = layer(input)
        print(result.shape)
        return result
    
    test_input = tf.random.uniform(shape=(1, 32, 32, 3))
    
    tmp_convop = ConvOps(32, 3, 1, 'same', 1)
    tmp_conv_result = experiment(test_input, tmp_convop)
    
    tmp_residual_plain = ResNetBlock(32, 3, 1, 1, 'same', 1)
    tmp_residual_plain_result = experiment(tmp_conv_result, tmp_residual_plain)
    
    tmp_residual_expand = ResNetBlockExpand(64, 3, 2, 1, 'same', 1)
    tmp_residual_expand_result = experiment(tmp_conv_result, tmp_residual_expand)
    
    tmp_residual_plain = ResNetBlock(32, 3, 1, 1, 'same', 4)
    tmp_residual_plain_result = experiment(tmp_conv_result, tmp_residual_plain)
    
    tmp_residual_expand = ResNetBlockExpand(64, 3, 2, 1, 'same', 4)
    tmp_residual_expand_result = experiment(tmp_conv_result, tmp_residual_expand)
    
    