import tensorflow_datasets as tfds


class TFDownloadDataset:
    def __init__(self, config):
        self.config = config
        assert self.config['from_tfds'] == True, 'The data_name is Invalid, please check data_config: data_name'
    
    def _download_data(self):
        examples, info = tfds.load(self.config['data_name'], as_supervised=True, with_info=True)
        return examples, info

    def _parse_data(self, examples):
        if len(examples) == 2:
            train_data = examples['train']
            valid_data = examples['validation']
            data = {'train': train_data,
                    'valid': valid_data}
            return data
        
        elif len(examples) == 3:
            train_data = examples['train']
            valid_data = examples['validation']
            test_data = examples['test']
            data = {'train':train_data,
                    'valid':valid_data,
                    'test':test_data}
            return data
    
    def get_data(self):
        examples, info = self._download_data()
        data = self._parse_data(examples)
        print(f'\n Parsed Data has {len(data)} parts')
        return data, info




if __name__ == "__main__":
    from utils.config_manager import ConfigManager

    config = ConfigManager('config/resnext').get_config()
    downlader = TFDownloadDataset(config)
    
    data, info = downlader.get_data()
