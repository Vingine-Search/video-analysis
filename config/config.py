from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import os
import json

config_file = os.path.join(os.path.dirname(__file__), 'config.json')

def reader():
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    return cfg

