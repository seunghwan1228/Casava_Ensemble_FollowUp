import tensorflow as tf
import numpy as np
import cv2
import albumentations as A
from preprocess.datadownlader import TFDownloadDataset

class Preprocessor:
    def __init__(self, config):
        self.config = config
        

    def resize_img(self, img):
        img = tf.image.resize(img, size=(self.config['image_height'],
                                         self.config['image_width']))
        return img
    

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
    

    def minmax_normalize_img(self, img):
        img = tf.cast(img, tf.float32)
        img = img / 255.
        return img
    
    # Augmentation functions

    def random_resize_crop_img(self, img):
        img = tf.image.resize(img, size=(self.config['image_height'] + 72,
                                        self.config['image_width'] + 72,
                                        ))
        img = tf.image.random_crop(img, size=(self.config['image_height'],
                                            self.config['image_width'],
                                            self.config['image_channel']))
        return img
    

    def random_horizontal_flip_img(self, img):
        img = tf.image.random_flip_up_down(img)
        return img
    

    def random_vertical_flip_img(self, img):
        img = tf.image.random_flip_left_right(img)
        return img


    def random_rotation_img(self, img):
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
        self.preprocessor = Preprocessor(config=self.config)
        
        if self.config['from_tfds']:
            self.datadownloader = TFDownloadDataset(config=self.config)
        
            self.data, self.info = self.datadownloader.get_data()
            
            if len(self.data) == 2:
                self.train_data = self.data['train']
                self.test_data = self.data['valid']
            
            else:
                self.train_data = self.data['train']
                self.valid_data = self.data['valid']
                self.test_data = self.data['test']
    
    def preprocess_data(self, img):
        img = self.preprocessor.tf_clahe_img(img)
        img = self.preprocessor.resize_img(img)
        img = self.preprocessor.random_horizontal_flip_img(img)
        img = self.preprocessor.random_vertical_flip_img(img)
        img = self.preprocessor.random_resize_crop_img(img)
        if self.config['normalizer'] == 'standard':
            img = self.preprocessor.standard_normalize_img(img)
        else:
            img = self.preprocessor.minmax_normalize_img(img)
            
        return img
    
    def tf_preprocess_data(self, img, target):
        img = self.preprocess_data(img)
        return (img, target)
    
    def get_train_data(self):
        pass
    
    def get_valid_data(self):
        pass
    
    def load_datasets(self):
        pass

    

if __name__ == "__main__":
    from preprocess.datadownlader import TFDownloadDataset
    from utils.config_manager import ConfigManager
    import matplotlib.pyplot as plt
    
    config = ConfigManager('config/resnext').get_config()
    tfdata = TFDownloadDataset(config=config)
    preprocessor = Preprocessor(config=config)
    
    dataset, info = tfdata.get_data()
    
    train = dataset['train']
    
    tmp_train = train.map(lambda x, y:preprocessor.resize_img(x))
    tmp_train = tmp_train.map(preprocessor.tf_clahe_img)
    tmp_train = tmp_train.map(preprocessor.standard_normalize_img)
    
    # ----------------------------- DATA LOADER Test ----------------- #
    data_loader = DataLoader(config=config)
    
    tmp_preprocessed = train.map(data_loader.tf_preprocess_data)
    
    tmp_preprocessed
    
    for sample in tmp_preprocessed.take(1):
        pass
    

    plt.imshow(preprocessor.de_standard_normalize_img(sample[0]))
    plt.show()

