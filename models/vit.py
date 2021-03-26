import tensorflow as tf
from models.vit_layers.transformer_layers import TransformerEncoder
from models.vit_layers.transformer_layers import PatchProjection

class VIT(tf.keras.models.Model):
    """[This Architecture is Plain Architecture
        This Architecture perform patch the image Inside of the architecture]
    """
    def __init__(self, config):
        super(VIT, self).__init__()
        self.config = config
        
        assert self.config['perform_patch'] == False, 'The config: perform_patch should be False'
        
        self.patch_info = tf.square(self.config['img_size'] // self.config['patch_size'])
        
        
        self.patch_projector = PatchProjection(input_img_size=self.config['img_size'], patch_size=self.config['patch_size'])
        
        self.projector = tf.keras.layers.Dense(self.config['model_dim'])   # b, n_p, dim

        # TODO: Requires to check the original implementation, Currently the pos-emb generate too large numbers
        
        self.pos_emb_constant = tf.keras.initializers.Constant(tf.range(0, self.patch_info, delta=1, dtype=tf.float32))
        self.pos_emb = self.add_weight(name='pos_emb', 
                                       shape=(1, self.patch_info, 1), 
                                       trainable=False, 
                                       dtype=tf.float32,
                                       initializer=self.pos_emb_constant)    # b, n_p, 1
        
        self.cls_emb = self.add_weight(name='cls_emb', shape=(1, 1), trainable=True, initializer='zero')
        
        
    def call(self, inputs):
        patch_img = self.patch_projector(inputs)
        patch_context = self.projector(patch_img)
        
        emb_patch_context = self.pos_emb + patch_context
        
        cls_emb = tf.broadcast_to(self.cls_emb, emb_patch_context.shape)
        concat_cls_emb = tf.concat([cls_emb, emb_patch_context], axis=1)
        
        return concat_cls_emb
    


if __name__ == "__main__":
    from utils.config_manager import ConfigManager
    
    config_manager = ConfigManager('config/vit')
    
    config = config_manager.get_config()
    tmp_model = VIT(config=config)
    

    tmp_input = tf.random.uniform(shape=(1, 224, 224, 3))
    
    test_output = tmp_model(tmp_input)
    
    
    test_output[0][0]