import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, default_level=logging.INFO):
    logging.basicConfig(filename=str(save_dir / 'logging.log'), level=default_level)
