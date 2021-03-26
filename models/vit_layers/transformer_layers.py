import tensorflow as tf
from models.vit_layers.transformer_utils import PatchProjection, MultiHeadAttention, DenseMLP, ConvMLP



class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, ff_dim, model_dim, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        
        self.norm_input = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(num_heads=num_heads, model_dim=model_dim)
        self.norm_mlp = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = DenseMLP(ff_dim=ff_dim, model_dim=model_dim)
        
    def call(self, inputs):
        """[summary]

        Args:
            inputs ([tensor]): [input tensor from 
                                image patched: [B, num_patchs, P**2*C] 
                              + patch embedding: [B, num_patchs + 1]]
        input -> layernorm -> mha -> + input -> layernorm -> mlp -> + mha_output 
        """
        input_ln = self.norm_input(inputs)
        mha_logit, mha_weight = self.mha(input_ln, input_ln, input_ln, mask=None)
        mha_addiction = inputs + mha_logit
        
        mha_ln = self.norm_mlp(mha_addiction)
        mlp_logit = self.mlp(mha_ln)
        mlp_addiction = mha_addiction + mlp_logit
        return mlp_addiction, mha_weight
    

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, ff_dim, model_dim, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        
        self.encoder_blocks = [TransformerEncoderBlock(num_heads, ff_dim, model_dim) for _ in range(num_layers)]
        
    def call(self, inputs):
        x = inputs
        attn_weight_dict = {}
        for enc_b_idx in range(len(self.encoder_blocks)):
            x, attn_weights = self.encoder_blocks[enc_b_idx](x)
            attn_weight_dict[f'Attention_Weight_layer_{enc_b_idx}'] = attn_weights
            
        return x, attn_weight_dict
             


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tmp_input = tf.random.uniform(shape=(2, 10, 20))
    # tmp_encoder = TransformerEncoderBlock(4, 16, 20)
    # tmp_output = tmp_encoder(tmp_input)
    # tmp_output[0]
    
    # tmp_output[1].shape
    # attn_checker = tmp_output[1][0]
    
    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(attn_checker[i, :, :])
        
    # plt.show()
    
    tmp_encoder = TransformerEncoder(2, 2, 10, 20)
    tmp_result = tmp_encoder(tmp_input)
    tmp_result
    tmp_result[1].keys()