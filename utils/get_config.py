import yaml
import os

# Xác định đường dẫn gốc của dự án
GET_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config(config_path='default.yaml'):
    with open(f'{GET_ROOT_PATH}/configs/{config_path}', 'r') as f:
        return yaml.safe_load(f)
