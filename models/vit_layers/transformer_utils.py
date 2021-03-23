import tensorflow as tf
import numpy as np



class PatchProjection(tf.keras.layers.Layer):
    def __init__(self, input_img_size, patch_size):
        super(PatchProjection, self).__init__()
        self.patch_size = patch_size
        self.input_img_size = input_img_size
        
        self.N = (self.input_img_size * self.input_img_size) / (self.patch_size ** 2)
        
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        img_patch = tf.image.extract_patches(inputs,
                                             sizes=[1, self.patch_size, self.patch_size, 1],
                                             strides=[1, self.patch_size, self.patch_size, 1],
                                             rates=[1, 1, 1, 1],
                                             padding='VALID')
        
        img_flatten = tf.reshape(img_patch, shape=(batch_size, self.N, -1))
        return img_flatten
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        
        assert model_dim % num_heads == 0
        
        self.depth = self.model_dim // self.num_heads
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        self.output_layer = tf.keras.layers.Dense(model_dim)
        
    def attention_score(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        
        logits = matmul_qk / dk
        if mask is not None:
            logits += (mask * -1e9)
        score = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(score, v)
        
        return output, score
    
    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # [B, H, seq_len, depth]
    
    def merge_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3]) # [B, seq_len, H, depth]
        return tf.reshape(x, shape=(batch_size, -1, self.model_dim))
    
    def call(self, q, k, v, mask):
        """[summary]

        Args:
            q ([type]): [Tensor: [B, seq_len_Q, Dim]]
            k ([type]): [Tensor: [B, seq_len_K, Dim]]
            v ([type]): [Tensor: [B, seq_len_K, Dim]]
            mask ([type]): [Tensor: [B, seq_len_Q]]

        Returns:
            [logits]: [Tensor: [B, seq_len_Q, dim]]
            [attention_weight]: [B, H, seq_len_Q, seq_len_K]
        """
        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)
        
        q_split = self.split_heads(Q)
        k_split = self.split_heads(K)
        v_split = self.split_heads(V)
        
        attn_logit, attn_weight = self.attention_score(q = q_split, 
                                                       k = k_split,
                                                       v = v_split,
                                                       mask=mask)
        merged_logit = self.merge_heads(attn_logit)
        output_logit = self.output_layer(merged_logit)
        
        return output_logit, attn_weight
    
class DenseMLP(tf.keras.layers.Layer):
    def __init__(self, ff_dim, model_dim, **kwargs):
        super(DenseMLP, self).__init__(**kwargs)
        self.ff_dim = ff_dim
        self.model_dim = model_dim
        self.ff_layer = tf.keras.layers.Dense(ff_dim, use_bias=False)
        self.output_layer = tf.keras.layers.Dense(model_dim, use_bias=False)
        
    def call(self, inputs):
        x = self.ff_layer(inputs)
        x = tf.nn.gelu(x)
        x = self.output_layer(x)
        return tf.nn.gelu(x)
    
class ConvMLP(tf.keras.layers.Layer):
    def __init__(self, ff_dim, model_dim, **kwargs):
        super(ConvMLP, self).__init__(**kwargs)
        self.ff_dim = ff_dim
        self.model_dim = model_dim
        self.ff_layer = tf.keras.layers.Conv2D(ff_dim, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.output_layer = tf.keras.layers.Conv2D(model_dim, kernle_size=3, strides=1, padding='same', use_bias=False)
    
    def call(self, inputs):
        x = self.ff_layer(inputs)
        x = tf.nn.gelu(x)
        x = self.output_layer(x)
        return tf.nn.gelu(x)
    
    


if __name__ == "__main__":
    tmp_input = tf.random.uniform(shape=[2, 3, 5])
    
    print(tmp_input)
    
    tmp_layer = MultiHeadAttention(4, 12)
    
    
    tmp_result_logit, tmp_result_weight = tmp_layer(tmp_input, tmp_input, tmp_input, None)
    
    
    tmp_result_logit.shape
    tmp_result_weight.shape
    
    tmp_layer.wq.weights