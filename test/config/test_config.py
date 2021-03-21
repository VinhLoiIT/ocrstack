from ocrstack.config.trainer import Config


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
