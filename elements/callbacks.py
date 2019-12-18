from abc import ABC
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import yaml
import pickle
import json
import numpy as np
from time import time
from multiprocessing import cpu_count
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from elements.selections import selection_functions
from elements.memetics import memetic_functions
from elements.mutations import mutation_functions
from elements.particle_swarm_iteration import swarm_iteration_functions


class CallbackBase(ABC):
    """Base class of callbacks for metaheuristic algorithms."""

    def __init__(self):
        self.model = None

    def on_search_begin(self, logs):
        pass

    def on_search_end(self, logs):
        pass

    def on_iteration_begin(self, logs):
        pass

    def on_iteration_end(self, logs):
        pass

    def on_step_begin(self, logs):
        pass

    def on_step_end(self, logs):
        pass

    def set_model(self, model):
        self.model = model


class LogToFile(CallbackBase):
    """Write log to a given file."""

    def __init__(self, log_dir="./logs"):
        super().__init__()

        self.file_path = log_dir

        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

    def on_iteration_end(self, logs):
        with open(os.path.join(self.file_path, "log"), "wb+") as log_file:
            pickle.dump(logs, log_file)

    def on_search_end(self, logs):
        with open(os.path.join(self.file_path, "log"), "wb+") as log_file:
            pickle.dump(logs, log_file)


class RemoteControl(CallbackBase):
    """Recompile the model from a given config file."""

    def __init__(self, config_file):
        super().__init__()

        self.config_file = config_file

    def on_iteration_end(self, logs):
        """Recompile the functions of the algorithm."""

        with open(self.config_file, 'r') as config_file:
            self.model.config = yaml.safe_load(config_file)

        if self.model.config["active"] is True:
            self.model.selection_function = selection_functions(**self.model.config)
            self.model.mutation_function = mutation_functions(**self.model.config)
            self.model.memetic_function = memetic_functions(**self.model.config)
            self.model.swarm_iteration_function = swarm_iteration_functions(**self.model.config)
            self.model.init_steps()

            self.model.stop = self.model.config["stop"]
            self.model.pool = self.model.config["pool"]
            self.model.pool_size = cpu_count() if self.model.config["pool_size"] is None else self.model.config[
                "pool_size"]
            variables = list(self.model.__dict__.keys())

            if "elitism" in variables:
                self.model.elitism = False if self.model.config["elitism"] is None else self.model.config["elitism"]
            if "num_of_new_individual" in variables:
                self.model.num_of_new_individual = self.model.population_size // 2 if self.model.config[
                                                                                          "num_of_new_individual"] is None else \
                self.model.config["num_of_new_individual"]
            if "num_of_crossover" in variables:
                self.model.num_of_crossover = self.model.population_size // 4 if self.model.config[
                                                                                     "num_of_crossover"] is None else \
                    self.model.config["num_of_crossover"]

            self.model.patience = float("inf") if self.model.config["patience"] is None else self.model.config[
                "patience"]
            self.model.max_iteration = float("inf") if self.model.config["max_iteration"] is None else \
                self.model.config["max_iteration"]
            self.model.max_fitness_eval = float("inf") if self.model.config["max_fitness_eval"] is None else \
                self.model.config["max_fitness_eval"]
            self.model.min_fitness = 0 if self.model.config["min_fitness"] is None else self.model.config["min_fitness"]


class SaveResult(CallbackBase):
    """Save the best and the global best individual in a given file."""

    def __init__(self, log_dir, iteration_end=False):
        super().__init__()

        self.result_file = os.path.join(log_dir, "result.txt")
        self.iteration_end = iteration_end

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    def save_result(self):
        result_dict = dict()
        best_individual_dict = self.model.best_individual()
        result_dict["best individual"] = best_individual_dict["best individual"][0]
        result_dict["global best individual"] = best_individual_dict["global best individual"][0]
        real_best_genes = self.model.fitness_function.genotype_to_phenotype(result_dict["best individual"])
        real_global_best_genes = self.model.fitness_function.genotype_to_phenotype(
            result_dict["global best individual"])
        result_dict["real_best_genes"] = real_best_genes
        result_dict["real_global_best_genes"] = real_global_best_genes

        for k, v in result_dict.items():
            result_dict[k] = v.tolist()

        with open(self.result_file, 'w+') as result_file:
            json.dump(result_dict, result_file)

        print("Bests save in file: ", self.result_file)

    def on_iteration_end(self, logs):
        if self.iteration_end:
            self.save_result()

    def on_search_end(self, logs):
        self.save_result()


class CheckPoint(CallbackBase):
    """Save the population in a given file."""

    def __init__(self, log_dir, only_last=True):
        super().__init__()

        self.checkpoint_file = os.path.join(log_dir, "chckpnt")
        self.only_last = only_last

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    # TODO: track config file
    def on_iteration_end(self, logs):
        if not self.only_last:
            file_name = self.checkpoint_file + "_{}".format(self.model.iteration)
        else:
            file_name = self.checkpoint_file

        with open(file_name, "wb+") as chckpnt_file:
            pickle.dump(self.model.population, chckpnt_file)

        print("Checkpoint save in file: ", file_name)


class DimReduction(CallbackBase):
    """
    Use TSNE dimension reduction on the population and visualize it.
    frequency: iteration frequency of dimension reduction
    dimension: n_component argument of TSNE
    perplexity: perplexity argument of TSNE
    """

    def __init__(self, log_dir, frequency, plot_runtime=False, dimensions=2, perplexity=30):
        super().__init__()

        self.save_dir = log_dir
        self.frequency = frequency
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.plot_rt = plot_runtime

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    def on_iteration_end(self, logs):
        if logs[-2]["iteration_end"]["iteration"] % self.frequency == 0:
            start = time()

            population = []
            fitness_values = []

            for individual in self.model.population:
                population.append(individual.genes)
                fitness_values.append(individual.fitness)

            population = np.array(population)
            fitness_values = np.array(fitness_values)

            embedded_population = TSNE(n_components=self.dimensions, perplexity=self.perplexity).fit_transform(
                population)

            if self.dimensions == 2:
                plt.scatter(embedded_population[:, 0], embedded_population[:, 1], c=fitness_values)
                plt.colorbar()
            elif self.dimensions == 3:
                fig = plt.figure()
                ax = Axes3D(fig)
                h = ax.scatter(embedded_population[:, 0], embedded_population[:, 1], embedded_population[:, 2],
                               c=fitness_values)

                fig.colorbar(h)

            plt.savefig(os.path.join(self.save_dir, '{}.png'.format(logs[-2]["iteration_end"]["iteration"])))

            if self.plot_rt:
                plt.show()

            plt.close()
            print("TSNE time: {0:.2f}s".format(time() - start))
