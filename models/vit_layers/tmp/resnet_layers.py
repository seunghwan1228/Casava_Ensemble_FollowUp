import tensorflow as tf
from models.resnet_layers.layers import ConvOps, ResNeXtBlockExpand, ResNetBlock, ResNetBlockExpand, ResNeXtBlock, ResNetBlockExpand



class ResNet50(tf.keras.Model):
    def __init__(self, config):
        super(ResNet50, self).__init__()
        self.config = config
        self.base_conv = ConvOps(filters=self.config['base_conv_filters'],
                                 kernel_size=(self.config['base_conv_kernel'], self.config['base_conv_kernel']),
                                 strides=(self.config['base_conv_strides'], self.config['base_conv_strides']),
                                 padding='same',
                                 groups=1)
        
        self.base_pool = tf.keras.layers.MaxPool2D(pool_size=(self.config['maxpool_size'], self.config['maxpool_size']),
                                                   strides=self.config['maxpool_stride'])
        
        self.block_1_exp = ResNeXtBlockExpand(conv_1_2_filters=self.config['conv1_1_2_filters'], 
                                              conv_3_filters=self.config['conv1_3_filters'],
                                              strides=1,
                                              groups=self.config['cardinality'])
        
        self.block_1 = [ResNeXtBlock(conv_1_2_filters=self.config['conv1_1_2_filters'], 
                                     conv_3_filters=self.config['conv1_3_filters'],
                                     groups=self.config['cardinality']) for _ in range(self.config['num_conv1_layers'] - 1)]
        
        
        self.block_2_exp = ResNeXtBlockExpand(conv_1_2_filters=self.config['conv2_1_2_filters'],
                                              conv_3_filters=self.config['conv2_3_filters'],
                                              strides=2,
                                              groups=self.config['cardinality'])
        self.block_2 = [ResNeXtBlock(conv_1_2_filters=self.config['conv2_1_2_filters'],
                                     conv_3_filters=self.config['conv2_3_filters'],
                                     groups=self.config['cardinality']) for _ in range(self.config['num_conv2_layers'] - 1)]
        
        
        self.block_3_exp = ResNeXtBlockExpand(conv_1_2_filters=self.config['conv3_1_2_filters'],
                                              conv_3_filters=self.config['conv3_3_filters'],
                                              strides=2,
                                              groups=self.config['cardinality'])
        self.block_3 = [ResNeXtBlock(conv_1_2_filters=self.config['conv3_1_2_filters'],
                                     conv_3_filters=self.config['conv3_3_filters'],
                                     groups=self.config['cardinality']) for _ in range(self.config['num_conv3_layers'] - 1)]
        
        self.block_4_exp = ResNeXtBlockExpand(conv_1_2_filters=self.config['conv4_1_2_filters'],
                                              conv_3_filters=self.config['conv4_3_filters'],
                                              strides=2,
                                              groups=self.config['cardinality'])

    def call(self, inputs):
        base_conv_result = self.base_conv(inputs)
        base_conv_pool = self.base_pool(base_conv_result)
        
        block_1_expanditure = self.block_1_exp(base_conv_pool)

        block_1_x = block_1_expanditure
        for block_1_layer in self.block_1:
            block_1_x = block_1_layer(block_1_x)
        
        
        block_2_expanditure = self.block_2_exp(block_1_x)
        block_2_x = block_2_expanditure
        for block_2_layer in self.block_2:
            block_2_x = block_2_layer(block_2_x)
        
        
        block_3_expanditure = self.block_3_exp(block_2_x)
        block_3_x = block_3_expanditure
        for block_3_layer in self.block_3:
            block_3_x = block_3_layer(block_3_x)
    
        return block_3_x

if __name__ == "__main__":
    from utils.config_manager import ConfigManager
    
    config_manager = ConfigManager('config/vit')
    
    config = config_manager.get_config()
    
    config_manager.print_config(config)
    tmp_input = tf.random.uniform(shape=(2, 224, 224, 3))
    tmp_model = ResNet50(config=config)
    tmp_output = tmp_model(tmp_input)
    
    print(tmp_output.shape)