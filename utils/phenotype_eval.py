import pickle
import argparse

from fitness_functions.fitness_function import FullyConnected, ConvNet


def print_result(dir_path):
    result_path = "/home/biot/projects/research/genetic_algorithm/logs/" + dir_path + "/result.txt"
    with open(result_path, 'r') as f:
        s = f.read()

    result_dict = eval(''.join(s.split("array")))

    print("Best individual\n")

    best_dict = result_dict.get("real_global_best_genes")
    fitness_value = result_dict.get("global best individual")[1]

    if dir_path[4:6] == "cn":
        result_dict = ConvNet.log_result(best_dict, fitness_value)
    else:
        result_dict = FullyConnected.log_result(best_dict, fitness_value)

    [print("{}: {}".format(key, best_dict.get(key))) for key in sorted(best_dict)]

    print("\nFitness value: {}\n".format(fitness_value))

    [print("{}: {}".format(key, result_dict.get(key))) for key in sorted(result_dict)]

    log_path = "/home/biot/projects/research/genetic_algorithm/logs/" + dir_path + "/log"
    with open(log_path, "rb") as log_file:
        logs = pickle.load(log_file)

    log_dict = dict()
    for log in logs:
        for key, inner_dict in log.items():
            for inner_key, item in inner_dict.items():
                if key + '/' + inner_key not in log_dict.keys():
                    log_dict[key + '/' + inner_key] = []
                log_dict[key + '/' + inner_key] += [item]

    keys = list(log_dict.keys())
    key_begins = list(set([key.split('/')[0] for key in keys]))
    key_begins.sort()

    [print("\n{}: {}".format(key, sum(log_dict.get(key)))) for key in sorted(log_dict) if key[-4:] in ["eval", "time"]]
    print()


def bayes_print(dir_path):
    result_path = "/home/biot/projects/research/genetic_algorithm/logs/" + dir_path + "/result.txt"

    with open(result_path, 'r') as f:
        s = f.read()

    result_dict = eval(''.join(s.split("array")))
    [print("{}: {}".format(key, result_dict.get(key))) for key in sorted(result_dict)]
    print()

    if result_dict.get("name") == "cn":
        parameter_dict = ConvNet.genotype_to_phenotype(genes=result_dict.get("best genes"))
    else:
        parameter_dict = FullyConnected.genotype_to_phenotype(genes=result_dict.get("best genes"))

    [print("{}: {}".format(key, parameter_dict.get(key))) for key in sorted(parameter_dict)]


if __name__ == "__main__":
    """python -m utils.phenotype_eval --dir_name gen-fc-19-04-08-17:45"""

    parser = argparse.ArgumentParser(description="Print phenotype, and evaluation log file.")

    parser.add_argument('--dir_name', help='Set the directory name of the log.', type=str)
    args = parser.parse_args()

    dir_name = args.dir_name

    if dir_name[:8] == "bayesian":
        bayes_print(dir_name)
    else:
        print_result(dir_name)

