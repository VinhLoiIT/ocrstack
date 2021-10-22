from typing import Any

import yaml


def load_yaml(path: str) -> Any:
    with open(path, 'rt') as f:
        return yaml.safe_load(f)
