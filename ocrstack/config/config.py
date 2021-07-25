from typing import Any, Dict, IO, Union
import yaml
from pathlib import Path


class Config(dict):

    def __init__(self):
        super(Config, self).__init__()

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        else:
            self[name] = Config()
            return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @staticmethod
    def from_yaml(f: Union[str, Path, IO]) -> 'Config':
        if isinstance(f, (str, Path)):
            with open(f, 'rt') as f:
                config_dict = yaml.safe_load(f)
        else:
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

    def to_yaml(self, f: Union[str, Path, IO]) -> None:
        def dump(f):
            yaml.safe_dump(self.to_dict(), f, indent=2)

        if isinstance(f, (str, Path)):
            with open(f, 'wt') as f:
                dump(f)
        else:
            dump(f)
