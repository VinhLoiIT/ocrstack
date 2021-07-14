from typing import Any, Dict
import yaml
from pathlib import Path


class Config(dict):

    def __init__(self):
        super(Config, self).__init__()

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @staticmethod
    def from_yaml(yaml_path: Path) -> 'Config':
        with open(yaml_path, 'rt') as f:
            config_dict = yaml.safe_load(f)
        return Config.from_dict(config_dict)

    @staticmethod
    def from_dict(d: Dict) -> 'Config':
        root = Config()
        for k, v in d.items():
            if isinstance(v, dict):
                v = Config.from_dict(v)
            root[k] = v
        return root

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self)
        for k, v in d.items():
            if isinstance(v, Config):
                v = v.to_dict()
            d[k] = v
        return d

    def to_yaml(self, yaml_path: Path) -> None:
        with open(yaml_path, 'wt') as f:
            yaml.safe_dump(self.to_dict(), f, indent=2)

    def __eq__(self, other: 'Config'):
        if len(self.keys()) != len(other.keys()):
            return False

        for k, v in self.items():
            if k not in other.keys():
                return False
            if v != other[k]:
                return False

        return True

    def __neq__(self, other: 'Config'):
        return not self == other
