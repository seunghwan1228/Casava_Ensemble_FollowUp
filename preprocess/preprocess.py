import tensorflow as tf
import numpy as np
import cv2
import albumentations as A
from preprocess.datadownlader import TFDownloadDataset

class Preprocess:
    def __init__(self, config):
        self.config = config
        
    @tf.function
    def resize_img(self, img):
        img = tf.image.resize(img, size=(self.config['image_height'],
                                         self.config['image_width']))
        return img
    
    @tf.function
    def standard_normalize_img(self, img):
        img = tf.cast(img, tf.float32)
        img = (img / 127.5) - 1
        return img
    
    @staticmethod
    def de_standard_normalize_img(img):
        """
        Denormalize as [-1, 1] to [0, 1]
        """
        return (img * 0.5) + 0.5
    
    @tf.function
    def minmax_normalize_img(self, img):
        img = tf.cast(img, tf.float32)
        img = img / 255.
        return img
    
    # Augmentation functions
    @tf.function
    def random_resize_crop_img(self, img):
        variable = tf.random.uniform(())
        if variable >= 0.5:
            img = tf.image.resize(img, size=(self.config['image_height'] + 72,
                                            self.config['image_width'] + 72))
            img = tf.image.random_crop(img, size=(self.config['image_height'],
                                                self.config['image_width']))
        return img
    
    @tf.function
    def random_horizontal_flip_img(self, img):
        variable = tf.random.uniform(())
        if variable >= 0.5:
            img = tf.image.random_flip_up_down(img)
        return img
    
    @tf.function
    def random_vertical_flip_img(self, img):
        variable = tf.random.uniform(())
        if variable >= 0.5:
            img = tf.image.random_flip_left_right(img)
        return img

    @tf.function
    def random_rotation_img(self, img):
        variable = tf.random.uniform(())
        if variable >= 0.5:
            rot_variable = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, rot_variable)
        return img
    
    def clahe_img(self, img):
        params = np.arange(3, 30, 3)
        select_param = np.random.choice(params, 1)
        select_param = int(select_param)
        transform = A.Compose([A.CLAHE(clip_limit=float(select_param), 
                                       tile_grid_size=(select_param, select_param))])
        img = transform(image=img)
        img = img['image']
        return tf.cast(img, tf.float32)

    def restore_clahe_img_shape(self, img):
        img.set_shape((self.config['image_height'] , self.config['image_width'], self.config['image_channel']))
        return img
    
    def tf_clahe_img(self, img):        
        img = tf.cast(img, tf.uint8)
        img = tf.numpy_function(func=self.clahe_img, inp=[img], Tout=tf.float32)
        img = self.restore_clahe_img_shape(img)
        return img
    

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.preprocessor = Preprocess(config=self.config)
        
        if self.config['from_tfds']:
            self.datadownloader = TFDownloadDataset(config=self.config)
        
            self.data, self.info = self.datadownloader
            
            if len(self.data) == 2:
                self.train_data = self.data['train']
                self.test_data = self.data['valid']
            
            else:
                self.train_data = self.data['train']
                self.valid_data = self.data['valid']
                self.test_data = self.data['test']
    
    def load_data(self):
        pass

    

if __name__ == "__main__":
    from preprocess.datadownlader import TFDownloadDataset
    from utils.config_manager import ConfigManager
    import matplotlib.pyplot as plt
    
    config = ConfigManager('config/resnext').get_config()
    tfdata = TFDownloadDataset(config=config)
    preprocessor = Preprocess(config=config)
    
    dataset, info = tfdata.get_data()
    
    train = dataset['train']
    
    tmp_train = train.map(lambda x, y:preprocessor.resize_img(x))
    tmp_train = tmp_train.map(preprocessor.tf_clahe_img)
    tmp_train = tmp_train.map(preprocessor.standard_normalize_img)
    
    
    for sample in tmp_train.take(1):
        pass
    plt.imshow(preprocessor.de_standard_normalize_img(sample))
    plt.show()

