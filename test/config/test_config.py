from io import StringIO
from ocrstack.config.config import Config


def test_compare_equal_config():
    conf_a = Config.from_dict({
        'param1': 1,
        'param2': 2.0,
        'param3': '3',
        'param4': [1, 2, 3, 4],
        'param5': {'param5_1': 1}
    })

    conf_b = Config.from_dict({
        'param1': 1,
        'param2': 2.0,
        'param3': '3',
        'param4': [1, 2, 3, 4],
        'param5': {'param5_1': 1}
    })

    assert conf_a == conf_b


def test_compare_not_equal_num_keys_config():
    conf_a = Config.from_dict({
        'param1': 1,
        'param2': 2.0,
        'param3': '3',
    })

    conf_b = Config.from_dict({
        'param1': 1,
        'param2': 2.0,
    })

    assert conf_a != conf_b


def test_compare_not_equal_config():
    conf_a = Config.from_dict({
        'param1': 1,
        'param2': 2.0,
        'param3': '3',
    })

    conf_b = Config.from_dict({
        'param1': 1,
        'param2': 2.0,
        'param3': '4',
    })

    assert conf_a != conf_b


def test_serialize_config():
    cfg = Config.from_dict({
        'param1': 1,
        'param2': 'x',
        'param3': Config.from_dict({
            'param3_1': 1.0,
            'param3_2': [1, 2, 3, 4],
        })
    })

    with StringIO() as f:
        cfg.to_yaml(f)
        f.seek(0)
        cfg2 = cfg.from_yaml(f)

    assert cfg == cfg2
