import configparser
import json

def type_convert(data):
    result = []
    for (key, value) in data:
        type_tag = key[:2]
        if type_tag == "s_":
            result.append((key[2:], value))
        elif type_tag == "f_":
            result.append((key[2:], float(value)))
        elif type_tag == "i_":
            result.append((key[2:], int(value)))
        elif type_tag == "b_":
            result.append((key[2:], bool(value)))
        else:
            raise ValueError('Invalid type tag "%s" found in ini file.' % type_tag)
    return result

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_values = {}
    for section in config.sections():
        config_values.update(dict(type_convert(config.items(section))))

    return config_values

def read_palette(palette_path):
    with open(palette_path, 'r') as json_file:
        palette_str_keys = json.load(json_file)

    palette = {tuple(map(int, k.strip('()').split(','))): v for k, v in palette_str_keys.items()}
    return palette
