import tensorflow as tf



class HybridExtractor(tf.keras.layers.Layer):
    def __init__(self, trainable=False):
        super(HybridExtractor, self).__init__()
        self.trainable = trainable
        self.resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        
        self.resnet_model = tf.keras.Model(self.resnet50.input, self.resnet50.get_layer('conv4_block6_out').output)
        self.resnet_model.trainable = self.trainable
        
    def call(self, inputs):
        return self.resnet_model(inputs)
        


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    tmp_input = tf.random.uniform(shape=(2, 224, 224, 3))
    plt.imshow(tmp_input[0])
    plt.show()
    
    tmp_hybrid = HybridExtractor()
    
    tmp_hybrid_result = tmp_hybrid(tmp_input)
    
    tmp_hybrid_result.shape
    
    # Check output
    
    tmp_result = tmp_hybrid_result[0]
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(tmp_result[:, :, i])
    plt.show()
    
