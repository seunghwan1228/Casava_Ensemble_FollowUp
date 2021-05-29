import tensorflow as tf
from models.vit_layers.transformer_layers import TransformerEncoder
from models.vit_layers.transformer_layers import PatchProjection
from models.vit_layers.transformer_utils import positional_encoding, DenseMLP

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
        # self.pos_emb_constant = tf.keras.initializers.Constant(tf.range(0, self.patch_info, delta=1, dtype=tf.float32))
        # self.pos_emb = self.add_weight(name='pos_emb', 
        #                                shape=(1, self.patch_info, 1), 
        #                                trainable=False, 
        #                                dtype=tf.float32,
        #                                initializer=self.pos_emb_constant)    # b, n_p, 1
        # From the original paer, they used sinusoidal embedding table
        self.pos_emb = positional_encoding(position=self.patch_info, d_model=self.config['model_dim'])
        
        self.cls_emb = self.add_weight(name='cls_emb', shape=(1, 1), trainable=True, initializer='zero') # (1, 1)
        
        self.mha_layer = TransformerEncoder(num_layers=self.config['num_layers'],
                                            num_heads=self.config['num_heads'],
                                             ff_dim = self.config['model_dim'],
                                             model_dim=self.config['model_dim'])
        
        self.mlp_layer = DenseMLP(ff_dim=self.config['model_dim'], model_dim=self.config['mlp_dim'])
        
        self.clf_output = tf.keras.layers.Dense(self.config['num_classes'])
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patch_img = self.patch_projector(inputs)
        patch_context = self.projector(patch_img)
        
        emb_patch_context = self.pos_emb + patch_context # b, 196, dim
        
        cls_emb = tf.broadcast_to(self.cls_emb, [batch_size, 1, self.config['model_dim']]) # b, 197, dim
        concat_cls_emb = tf.concat([cls_emb, emb_patch_context], axis=1)
        
        attn_logit, attn_weight_dict = self.mha_layer(concat_cls_emb)
        
        mlp_output = self.mlp_layer(attn_logit)
        
        if self.config['pool_by_slice']:
            mlp_pool_logit = mlp_output[:, 0]
        else:
            mlp_pool_logit = tf.reduce_mean(mlp_output, axis=1)
        
        clf_logit = self.clf_output(mlp_pool_logit)
        
        return clf_logit
    


if __name__ == "__main__":
    from utils.config_manager import ConfigManager
    
    config_manager = ConfigManager('config/vit')
    
    config = config_manager.get_config()
    tmp_model = VIT(config=config)
    
    tmp_model.patch_info

    tmp_input = tf.random.uniform(shape=(1, 224, 224, 3))
    
    test_output = tmp_model(tmp_input)
    
    

    test_output