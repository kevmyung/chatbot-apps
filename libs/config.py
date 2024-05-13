import os
import yaml

def load_model_config():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(file_dir, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config['models']

def load_language_config(language):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(file_dir, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config['languages'][language]

