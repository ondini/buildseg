import torch
import numpy as np
import json
from pathlib import Path
from collections import OrderedDict

def fix_seed(seed):
    np.random.seed(seed)     
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_json(filepath):
    filepath = Path(filepath)
    with filepath.open('rt') as fhandle:
        return json.load(fhandle)

def write_json(content, filepath):
    filepath = Path(filepath)
    with filepath.open('wt') as fhandle:
        json.dump(content, fhandle, sort_keys=False)
