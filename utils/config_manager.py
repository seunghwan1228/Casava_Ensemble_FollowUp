from pathlib import Path
import ruamel.yaml
import os
from datetime import datetime


class ConfigManager:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self._yaml_loader = ruamel.yaml.YAML()

    def _read_config(self, config_file):
        with open(config_file, 'rb') as cfg:
            config_dict = self._yaml_loader.load(cfg)
        return config_dict

    def _load_multiple_config(self):
        config_file_list = [self.config_path / Path(cfg_file)  for cfg_file in os.listdir(self.config_path) if cfg_file.endswith('.yaml')]
        return config_file_list

    def _merge_config_dict(self, list_config_dict):
        config_dict = {}
        for cfg_file in list_config_dict:
            loaded_config = self._read_config(cfg_file)
            config_dict.update(loaded_config)
        return config_dict

    def get_config(self):
        config_list = self._load_multiple_config()
        configs = self._merge_config_dict(config_list)
        return configs

    def print_config(self, full_config_dict):
        print('\nCONFIGURATION\n')
        for key, value in full_config_dict.items():
            print(f' - {key}: {value}')
    
    def _update_config(self, full_config_dict):
        full_config_dict['experiment_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        return full_config_dict

    def dump_config(self, full_config_dict, model_name):
        full_config_dict = self._update_config(full_config_dict)
        os.makedirs((Path(full_config_dict['log_directory']) / model_name), exist_ok=True)
        with open(Path(full_config_dict['log_directory']) /model_name/'configs.yaml', 'w') as cfg_writer:
            self._yaml_loader.dump(full_config_dict, cfg_writer)
    
    




if __name__ == "__main__":
    tmp_config = ConfigManager('config/resnext')
    tmp_list = tmp_config._load_multiple_config()
    tmp_full_loaded = tmp_config._merge_config_dict(tmp_list)
    print(tmp_full_loaded)
    tmp_config.print_config(tmp_full_loaded)
    tmp_config.dump_config(tmp_full_loaded, 'vit')
