import yaml
import os
import datetime
import argparse

from fitness_functions.fitness_function import RastriginFunction, FullyConnected, ConvNet, TriangleDraw, CircleDraw
from elements.callbacks import LogToFile, RemoteControl, SaveResult, CheckPoint, DimReduction
from algorithms.pso_alg import ParticleSwarm
from algorithms.gen_alg import GeneticAlgorithm

fitness_func_dict = {"rf": RastriginFunction,
                     "fc": FullyConnected,
                     "cn": ConvNet,
                     "td": TriangleDraw,
                     "cd": CircleDraw,
                     }

algorithm_dict = {"gen": GeneticAlgorithm,
                  "pso": ParticleSwarm,
                  }


def get_path(parameters):
    if parameters["dir_path"]:
        path = parameters.path
    else:
        if parameters["load"]:
            path = "./logs/{}-{}-{}-load".format(parameters["type"], parameters["fitness_function"],
                                                 datetime.datetime.now().strftime('%y-%m-%d-%H:%M'))
        else:
            path = "./logs/{}-{}-{}".format(parameters["type"], parameters["fitness_function"],
                                            datetime.datetime.now().strftime('%y-%m-%d-%H:%M'))
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_config(parameters, path):
    if parameters["config_path"]:
        with open(parameters["config_path"], 'r') as config_file:
            config = yaml.safe_load(config_file)
    else:
        with open("config_tmp.yml", 'r') as config_file:
            config = yaml.safe_load(config_file)

    config_path = os.path.join(path, "config.yml")
    with open(config_path, "w+") as log_file:
        yaml.dump(config, log_file, default_flow_style=False)

    return config, config_path


def get_callback_list(parameters, path, config_path):
    callback_dict = {"ltf": lambda: LogToFile(log_dir=path),
                     "rc": lambda: RemoteControl(config_file=config_path),
                     "cp": lambda: SaveResult(log_dir=path, iteration_end=True),
                     "sr": lambda: CheckPoint(log_dir=path, only_last=True),
                     "dr": lambda: DimReduction(log_dir=path, dimensions=2, frequency=2, plot_runtime=False),
                     }

    callback_list = []
    for cb in parameters["callbacks"]:
        callback_list.append(callback_dict[cb]())

    if parameters["all_cb"]:
        for callback in callback_dict.values():
            callback_list.append(callback())

    return callback_list


def configure_algorithm(parameters):
    path = get_path(parameters)

    config, config_path = get_config(parameters, path)

    # TODO: **config --> config object
    alg = algorithm_dict[parameters["type"]](**config)

    fitness_func = fitness_func_dict[parameters["fitness_function"]]()

    callback_list = get_callback_list(parameters, path, config_path)

    if parameters["load"]:
        alg.load_population(checkpoint_path=parameters["load"])

    alg.compile(config=config, fitness_function=fitness_func, callbacks=callback_list)

    return alg


# TODO: show result, dimred with const metric, graphic_config, messy genetic, pause/continue


if __name__ == '__main__':
    """python app_alg.py --type gen --ff fc --all_cb"""

    parser = argparse.ArgumentParser(description="Compile and run a metaheuristic  algorithm.")

    parser.add_argument('--dir_path', help='Dir path for logs, checkpoint and result files.', type=str)
    parser.add_argument('--type', help='Type of algorithm.', choices=['gen', 'pso'], type=str)
    parser.add_argument('--load', help='File path to checkpoint path to load.', type=str)
    parser.add_argument('--config_path', help='File path to config file.', type=str)
    parser.add_argument('--callbacks',
                        help='List of callbacks to use. ltf: LogToFile, rc: RemoteControl,'
                             ' sr: SaveResult, cp: CheckPoint, dr: DimReduction',
                        default=[], type=str, nargs='+')
    parser.add_argument('--ff', help='Set the fitness function. rf: Rastrigin, fc: FullyConnected, cn: ConvNet',
                        choices=['rf', 'fc', 'cn', 'td', 'cd'], default='rf', type=str)
    parser.add_argument('--all_cb', help='Use all of the callbacks.', action='store_true')

    # Parse the input parameters to the script
    args = parser.parse_args()
    parameter_dict = {"dir_path": args.dir_path,
                      "load": args.load,
                      "type": args.type,
                      "config_path": args.config_path,
                      "callbacks": args.callbacks,
                      "fitness_function": args.ff,
                      "all_cb": args.all_cb
                      }

    algorithm = configure_algorithm(parameter_dict)

    print("Run algorithm\n")
    algorithm.run()
