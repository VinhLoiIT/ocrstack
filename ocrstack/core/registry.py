# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) OCRStack. We changed the original file.
# pyre-ignore-all-errors[2,3]
from typing import Any, Dict, Iterable, Iterator, Tuple

from tabulate import tabulate


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: Any = None, prefix: str = '') -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = prefix + func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = prefix + obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    def _make_instance(self, cfg: Dict):
        args = cfg.get('args', None) or {}
        return self.get(cfg['name'])(**args)

    def build_from_cfg(self, cfg: Dict) -> Any:
        if isinstance(cfg, list):
            return [self.build_from_cfg(item) for item in cfg]
        if isinstance(cfg, Dict):
            if 'name' in cfg.keys():
                args = cfg.get('args', {}) or {}
                args = self.build_from_cfg(args)
                if isinstance(args, list):
                    return self.get(cfg['name'])(*args)
                if isinstance(args, dict):
                    return self.get(cfg['name'])(**args)
            else:
                return {
                    k: self.build_from_cfg(v) for k, v in cfg.items()
                }

        return cfg

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__
