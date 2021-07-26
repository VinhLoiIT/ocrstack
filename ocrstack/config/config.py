from io import StringIO
from pathlib import Path
from typing import IO, Any, Dict, List, Tuple, Union

import yaml


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

    def __str__(self) -> str:
        def flatten(d, prefix='') -> List[Tuple[str, Any]]:
            items = []
            for k, v in d.items():
                new_key = f'{prefix}.{k}' if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key))
                else:
                    items.append((new_key, v))
            return items

        cfg_list = flatten(self, 'cfg')
        max_length = max([len(item[0]) for item in cfg_list])
        with StringIO() as f:
            print('*' * (max_length * 2 + 3), file=f)
            for k, v in cfg_list:
                print(str.ljust(k, max_length), '*', v, file=f)
            print('*' * (max_length * 2 + 3), file=f)
            f.seek(0)
            return f.read()
