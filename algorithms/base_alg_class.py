from abc import ABC, abstractmethod
import time
import pickle
from functools import partial
from pdb import set_trace

from elements.selections import selection_functions
from elements.memetics import memetic_functions
from elements.mutations import mutation_functions
from elements.particle_swarm_iteration import swarm_iteration_functions


class BaseAlgorithmClass(ABC):
    """
    Base class of metaheuristic algoritms.

    TODO: How to use
    """

    def __init__(self,
                 population_size=50,
                 chromosome_size=10,
                 max_iteration=None,
                 max_fitness_eval=None,
                 min_fitness=None,
                 patience=None,
                 pool_size=None,
                 pool=False):

        self.patience = float("inf") if patience is None else patience
        self.max_iteration = float("inf") if max_iteration is None else max_iteration
        self.max_fitness_eval = float("inf") if max_fitness_eval is None else max_fitness_eval
        self.min_fitness = 0 if min_fitness is None else min_fitness
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.pool_size = pool_size
        self.pool = pool

        self.population = None

        self.fitness_function = None
        self.selection_function = None
        self.mutation_function = None
        self.swarm_iteration_function = None
        self.memetic_function = None

        self.config = None
        self.callbacks = None
        self.logs = [dict()]

        self.iteration_steps = []
        self.iteration = 0
        self.no_improvement = 0
        self.num_of_fitness_eval = 0
        self.best_fitness = None
        self.stop = False
        self.progress_bar = True

        if self.pool:
            print('Using process pool with pool size {}.\n'.format(self.pool_size))

    def compile(self, config, fitness_function, callbacks=None):
        """Compile the functions of the algorithm."""

        self.config = config

        # print config
        for key, item in config.items():
            if item is True:
                print("{0}: {1}".format(key, item))
        print('\n')

        for key, item in config.items():
            if isinstance(item, str) and item is not True and item:
                print("{0}: {1}".format(key, item))
        print('\n')

        for key, item in config.items():
            if isinstance(item, (int, float)) and item is not True and item:
                print("{0}: {1}".format(key, item))
        print('\n')

        self.fitness_function = fitness_function
        self.callbacks = callbacks if callbacks else []

        self.selection_function = selection_functions(**self.config)
        self.mutation_function = mutation_functions(**self.config)
        self.swarm_iteration_function = swarm_iteration_functions(**self.config)
        self.memetic_function = memetic_functions(**self.config)

        self.init_population()

    def init_population(self):
        """
        Create and initialize the population.
        Calculate the population's fitness and rank the population
        by fitness ascending order.
        """
        print("Initialize population")

        self.create_population()
        self.population.rank_population()
        self.population.init_global_and_personal_best()
        self.init_steps()
        self.get_all_fitness_eval()

    @abstractmethod
    def create_population(self):
        """Set the appropriate population type."""
        pass

    @abstractmethod
    def init_steps(self):
        """Initialize the iteration steps."""
        pass

    def next_iteration(self):
        """Create the next iteration or generation with the corresponding steps."""

        self.callbacks_on_iteration_begin()

        for step in self.iteration_steps:
            self.callbacks_on_step_begin()
            best_fitness_before_step = self.population.get_best_fitness()
            step_time, name = step()

            self.rank_population()
            sum_of_eval = self.get_all_fitness_eval()

            self.set_personal_bests()
            self.set_global_best()

            self.print_best_values(top_n=4)
            self.step_update_log(name, step_time, sum_of_eval, best_fitness_before_step)

            self.callbacks_on_step_end()

        self.cut_pop_size()

    def run(self):
        """Run (solve) the algorithm."""

        self.iteration = 1
        self.best_fitness = self.population.get_best_fitness()
        self.callbacks_on_search_begin()

        try:
            while self.no_improvement < self.patience and self.max_iteration >= self.iteration \
                    and self.min_fitness < self.best_fitness and self.max_fitness_eval > self.num_of_fitness_eval \
                    and self.stop is False:
                start = time.time()
                print('*' * 36, '{}. iteration'.format(self.iteration), '*' * 36)
                best_fitness_before_iteration = self.population.get_best_fitness()
                self.next_iteration()

                iteration_time = time.time() - start
                print('Number of fitness evaluation so far: ', self.num_of_fitness_eval)
                print('Iteration process time: {0:.2f}s\n\n'.format(iteration_time))

                if self.best_fitness > self.population.get_global_best().fitness:
                    self.no_improvement = 0
                    self.best_fitness = self.population.get_global_best().fitness
                else:
                    self.no_improvement += 1

                self.iteration_update_log(iteration_time, best_fitness_before_iteration)
                self.callbacks_on_iteration_end()

                self.iteration += 1
        except KeyboardInterrupt:
            # in case of Ctrl + c the self.callbacks_on_search_end() should be called
            pass

        self.callbacks_on_search_end()

    def step_update_log(self, name, step_time, sum_of_eval, best_fitness_before_step):
        """Update self.logs on step end."""

        best_fitness = self.population.get_best_fitness()
        improvement = - (best_fitness - best_fitness_before_step)

        self.logs[-1][name] = {"iteration": self.iteration,
                               "step_time": step_time,
                               "sum_of_eval": sum_of_eval,
                               "improvement": improvement}

    def iteration_update_log(self, step_time, best_fitness_before_iteration):
        """Update self.logs on iter end."""
        best_fitness = self.population.get_best_fitness()
        improvement = - (best_fitness - best_fitness_before_iteration)

        if self.iteration == 1:
            sum_of_eval = self.num_of_fitness_eval
        else:
            sum_of_eval = self.num_of_fitness_eval - sum(
                self.logs[i]["iteration_end"]["sum_of_eval"] for i in range(len(self.logs) - 1))

        self.logs[-1]["iteration_end"] = {"iteration": self.iteration,
                                          "step_time": step_time,
                                          "sum_of_eval": sum_of_eval,
                                          "best_fitness": best_fitness,
                                          "global_best_fitness": self.population.get_global_best().fitness,
                                          "improvement": improvement}
        self.logs.append(dict())

    def callbacks_on_search_begin(self):
        """Call the on_search_begin function of the callbacks."""
        for callback in self.callbacks:
            callback.on_search_begin(self.logs)
            callback.set_model(self)

    def callbacks_on_search_end(self):
        """Call the on_search_end function of the callbacks."""
        for callback in self.callbacks:
            callback.on_search_end(self.logs)

    def callbacks_on_iteration_begin(self):
        """Call the on_iteration_begin function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_begin(self.logs)

    def callbacks_on_iteration_end(self):
        """Call the on_iteration_end function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.logs)

    def callbacks_on_step_begin(self):
        """Call the on_step_begin function of the callbacks."""
        for callback in self.callbacks:
            callback.on_step_begin(self.logs)

    def callbacks_on_step_end(self):
        """Call the on_step_end function of the callbacks."""
        for callback in self.callbacks:
            callback.on_step_end(self.logs)

    def last_iteration(self):
        """Return members of the last iteration as a generator function."""
        return ((member.fitness, member.genes) for member in self.population.get_all())

    def best_individual(self):
        """Return the individual with the best fitness in the current generation."""
        best_genes = self.population.get_best_genes()
        best_fitness = self.population.get_best_fitness()
        return {"best individual": (best_genes, best_fitness), "global best individual": (
        self.population.get_global_best().genes, self.population.get_global_best().fitness)}

    def rank_population(self):
        """Sort the population by fitness ascending order."""
        self.population.rank_population()

    def set_global_best(self):
        """Set global best values."""
        self.population.set_global_best()

    def set_personal_bests(self):
        """Set personal bests of the parcticles."""
        self.population.set_personal_bests()

    def cut_pop_size(self):
        """Resize the current population to pop size."""
        self.population.cut_pop_size()

    def add_individual_to_pop(self, individual):
        """Add an individual to the current population."""
        self.population.add_individual_to_pop(individual)

    def add_new_individual(self):
        """Add new individual to the current population."""
        self.population.add_new_individual()

    def get_all_fitness_eval(self):
        """Count all evaluation and set they to 0."""
        sum_of_eval = 0
        for i, member in enumerate(self.population):
            sum_of_eval += member.num_fitness_eval
            member.num_fitness_eval = 0

        self.num_of_fitness_eval += sum_of_eval
        return sum_of_eval

    def print_best_values(self, top_n=4):
        """Print the top n pieces fitness values"""
        print('Best fitness values:')
        for i in range(top_n if self.population_size >= top_n else self.population_size):
            print('{0:.3f}'.format(self.population[i].fitness))
        print('Global best fitness value:')
        print('{0:.3f}'.format(self.population.get_global_best().fitness))
        print('\n')

    def load_population(self, checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as log_file:
                population = pickle.load(log_file)
        except FileNotFoundError:
            print("Wrong file path!")
            exit()

        assert population.pop_size == self.population_size
        assert population.chromosome_size == self.chromosome_size

        self.population = population
        print('Load population from {}.'.format(checkpoint_path))
