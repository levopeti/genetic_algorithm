import json
import yaml
import os
from termcolor import colored

config_dict = dict()
config_path = "NoSet"


def init_from_dict(dictionary):
    for k, v, in dictionary.items():
        log = {"iteration": 0, "value": v}
        config_dict[k] = [log]


def update_config(k, v, iteration):
    log = {"iteration": iteration, "value": v}
    config_dict[k].append(log)


def load_from_json(json_path):
    global config_dict
    with open(json_path) as json_file:
        config_dict = json.load(json_file)

    print(colored("Config json: ", 'green') + "Config json loaded from file: ", json_path)


def save_to_json():
    if config_path == "NoSet":
        print(colored("Config json: ", 'red') + "There is no path to save config json.")

    with open(config_path, 'w+') as save_file:
        json.dump(config_dict, save_file)

    print(colored("Config json: ", 'green') + "Config json saved in file: ", config_path)


def load_from_yaml(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        config_yaml_dict = yaml.safe_load(yaml_file)
    init_from_dict(config_yaml_dict)


def set_config_path(dir_path):
    global config_path
    config_path = os.path.join(dir_path, "config.json")

