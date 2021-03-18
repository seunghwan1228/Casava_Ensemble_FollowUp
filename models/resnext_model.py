import tensorflow as tf
from models.resnet_layers.component_layers import ResnextBlock, ResNextSumBlock, ResNextConcatBlock

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
        
        self.conv_1_bottle_neck = tf.keras.layers.Conv2D(self.config['conv2_3_filters'],
                                                         kernel_size=1,
                                                         strides=1, 
                                                         padding='same',
                                                         activation='relu')
        
        self.conv_2_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv2_1_2_filters'],
                                                 conv_3_filters=self.config['conv2_3_filters'], 
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv2_layers'])]

        self.conv_2_bottle_neck = tf.keras.layers.Conv2D(self.config['conv3_3_filters'],
                                                         kernel_size=1,
                                                         strides=1, 
                                                         padding='same',
                                                         activation='relu')   

        self.conv_3_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv3_1_2_filters'],
                                                 conv_3_filters=self.config['conv3_3_filters'],
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv3_layers'])]
        
        self.conv_3_bottle_neck = tf.keras.layers.Conv2D(self.config['conv4_3_filters'],
                                                         kernel_size=1,
                                                         strides=1, 
                                                         padding='same',
                                                         activation='relu')   
        
        self.conv_4_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv4_1_2_filters'],
                                                 conv_3_filters=self.config['conv4_3_filters'],
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv4_layers'])]
        
        self.conv_4_bottle_neck = tf.keras.layers.Conv2D(self.config['conv5_3_filters'],
                                                         kernel_size=1,
                                                         strides=1, 
                                                         padding='same',
                                                         activation='relu')  
        
        self.conv_5_blocks = [ResNextConcatBlock(conv_1_2_filters=self.config['conv5_1_2_filters'],
                                                 conv_3_filters=self.config['conv5_3_filters'],
                                                 cardinality=self.config['cardinality']) for _ in range(self.config['num_conv5_layers'])]

        
        self.ga_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(config['num_classes'])
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pool(x)
        x = self.conv_1_bottle_neck(x)
        
        conv_1_input = x
        for conv2_layer in self.conv_2_blocks:
            conv_2_result = conv2_layer(conv_1_input)
        
        conv_2_input = self.conv_2_bottle_neck(conv_2_result)
        for conv3_layer in self.conv_3_blocks:
            conv_3_result = conv3_layer(conv_2_input)
            
        conv_3_input = self.conv_3_bottle_neck(conv_3_result)
        for conv4_layer in self.conv_4_blocks:
            conv_4_result = conv4_layer(conv_3_input)
            
        conv_4_input = self.conv_4_bottle_neck(conv_4_result)
        for conv5_layer in self.conv_5_blocks:
            x = conv5_layer(conv_4_input)
            
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
        
        self.conv_1_bottle_neck = tf.keras.layers.Conv2D(self.config['conv2_3_filters'],
                                                         kernel_size=1,
                                                         strides=1, 
                                                         padding='same',
                                                         activation='relu')
        
        self.conv_2_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv2_1_2_filters'],
                                              conv_3_filters=self.config['conv2_3_filters'], 
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv2_layers'])]
        
        self.conv_2_bottle_neck = tf.keras.layers.Conv2D(self.config['conv3_3_filters'],
                                                    kernel_size=1,
                                                    strides=1, 
                                                    padding='same',
                                                    activation='relu')        

        self.conv_3_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv3_1_2_filters'],
                                              conv_3_filters=self.config['conv3_3_filters'],
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv3_layers'])]
        
        self.conv_3_bottle_neck = tf.keras.layers.Conv2D(self.config['conv4_3_filters'],
                                                         kernel_size=1,
                                                         strides=1, 
                                                         padding='same',
                                                         activation='relu')    
         
        self.conv_4_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv4_1_2_filters'],
                                              conv_3_filters=self.config['conv4_3_filters'],
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv4_layers'])]

        self.conv_4_bottle_neck = tf.keras.layers.Conv2D(self.config['conv5_3_filters'],
                                                    kernel_size=1,
                                                    strides=1, 
                                                    padding='same',
                                                    activation='relu')  
        
        self.conv_5_blocks = [ResNextSumBlock(conv_1_2_filters=self.config['conv5_1_2_filters'],
                                              conv_3_filters=self.config['conv5_3_filters'],
                                              cardinality=self.config['cardinality']) for _ in range(self.config['num_conv5_layers'])]
    
        self.ga_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(config['num_classes'])
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pool(x)
        x = self.conv_1_bottle_neck(x)
        
        conv_1_input = x
        for conv2_layer in self.conv_2_blocks:
            conv_2_result = conv2_layer(conv_1_input)
        
        conv_2_input = self.conv_2_bottle_neck(conv_2_result)
        for conv3_layer in self.conv_3_blocks:
            conv_3_result = conv3_layer(conv_2_input)
            
        conv_3_input = self.conv_3_bottle_neck(conv_3_result)
        for conv4_layer in self.conv_4_blocks:
            conv_4_result = conv4_layer(conv_3_input)
            
        conv_4_input = self.conv_4_bottle_neck(conv_4_result)
        for conv5_layer in self.conv_5_blocks:
            x = conv5_layer(conv_4_input)
            
        x = self.ga_pool(x)
        return self.output_layer(x)