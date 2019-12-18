import time
from functools import partial

from pathos.multiprocessing import Pool
from multiprocessing import cpu_count, Manager
from progressbar import ProgressBar, Bar, Percentage, ETA

from elements.population import Population
from algorithms.base_alg_class import BaseAlgorithmClass


class GeneticAlgorithm(BaseAlgorithmClass):
    """
    Genetic Algorithm class.
    This is the main class that controls the functionality of the Genetic Algorithm.
    """

    def __init__(self,
                 population_size=50,
                 chromosome_size=10,
                 max_iteration=None,
                 max_fitness_eval=None,
                 min_fitness=None,
                 patience=None,
                 pool_size=cpu_count(),
                 pool=False,
                 num_of_new_individual=None,
                 num_of_crossover=None,
                 elitism=False,
                 *args,
                 **kwargs):
        super().__init__(population_size=population_size,
                         chromosome_size=chromosome_size,
                         max_iteration=max_iteration,
                         max_fitness_eval=max_fitness_eval,
                         min_fitness=min_fitness,
                         patience=patience,
                         pool_size=pool_size,
                         pool=pool)

        self.num_of_new_individual = self.population_size // 2 if num_of_new_individual is None else num_of_new_individual
        self.num_of_crossover = self.population_size // 4 if num_of_crossover is None else num_of_crossover
        self.elitism = elitism

    def create_population(self):
        """
        Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        if self.population is None:
            self.population = Population(self.population_size, self.chromosome_size, self.fitness_function, self.pool,
                                         self.pool_size)

    def init_steps(self):
        """Initialize the iteration steps."""
        self.iteration_steps = []

        if self.config["crossover"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Crossover"))

        if self.config["differential_evolution"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Differential evolution"))

        if self.config["invasive_weed"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Invasive weed"))

        if self.config["add_pure_new"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Add pure new"))

        if self.config["mutation"]:
            self.iteration_steps.append(partial(self.modify_one_by_one_function, "Mutation"))

        if self.config["memetic"]:
            self.iteration_steps.append(partial(self.modify_one_by_one_function, "Local search"))

    def add_new_individuals_function(self, name):
        """
        Add new individuals to the population with a given method
        (crossover, differential evolution, invasive weed, add pure individual).
        """
        start = time.time()

        if name == "Crossover":
            current_function = partial(self.population.crossover, self.selection_function)
            name = name[:9] + ' ' + self.config["selection_type"]
            iterator = range(self.num_of_crossover)
        elif name == "Differential evolution":
            current_function = partial(self.population.differential_evolution, self.config["CR"], self.config["F"])
            iterator = range(len(self.population))
        elif name == "Invasive weed":
            current_function = partial(self.population.invasive_weed, self.iteration, self.config["iter_max"],
                                       self.config["e"], self.config["sigma_init"], self.config["sigma_fin"],
                                       self.config["N_min"], self.config["N_max"])
            iterator = self.population[:]
        elif name == "Add pure new":
            current_function = self.population.add_new_individual
            iterator = range(self.num_of_new_individual)
        else:
            raise NameError("Bad type of function.")
        if self.pool:
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)

            def pool_function(inside_lock, inside_counter, inside_member):
                inside_lock.acquire()
                inside_counter.value += 1
                inside_lock.release()

                inside_members = current_function(inside_member, gpu=inside_counter.value % 4)
                return inside_members

            func = partial(pool_function, lock, counter)

            members = p.map(func, iterator)

            if name == "Differential evolution":
                self.population.current_population = members
            else:
                try:
                    members = sum(members, [])
                except TypeError:
                    pass

                for member in members:
                    self.population.add_individual_to_pop(member)
            p.terminate()
        else:
            members = []
            for argument in iterator:
                member = current_function(argument, gpu=0)
                members.append(member)

            if name == "Differential evolution":
                self.population.current_population = members
            else:
                try:
                    members = sum(members, [])
                except TypeError:
                    pass

                for member in members:
                    self.population.add_individual_to_pop(member)

        step_time = time.time() - start

        if step_time < 120:
            print('{0} time: {1:.2f}s\n'.format(name, step_time))
        else:
            print('{0} time: {1:.2f}min\n'.format(name, step_time // 60))

        return step_time, name

    def modify_one_by_one_function(self, name):
        """Apply a function (local search, mutation) to all chromosomes."""
        start = time.time()
        if self.progress_bar:
            print("{}:".format(name))

        if name == "Local search":
            current_function = self.memetic_function
        elif name == "Mutation":
            current_function = self.mutation_function
            name = name[:8] + ' ' + self.config["mutation_type"]
        else:
            raise NameError("Bad type of function.")

        if self.iteration > 1:
            if name in self.logs[-2].keys():
                if self.logs[-2][name]["step_time"] < 4:
                    self.progress_bar = False
                else:
                    self.progress_bar = True

        if self.fitness_function.name in ["fully connected", "convnet"]:
            self.progress_bar = False

        if self.pool:
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)
            if self.progress_bar:
                pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], term_width=60, maxval=len(self.population)).start()
            else:
                pbar = None

            def pool_function(inside_lock, inside_counter, inside_member):
                inside_lock.acquire()
                inside_counter.value += 1
                inside_lock.release()

                inside_member.apply_on_chromosome(current_function, gpu=inside_counter.value % 4)

                inside_lock.acquire()
                if pbar:
                    pbar.update(inside_counter.value)
                inside_lock.release()

                return inside_member

            func = partial(pool_function, lock, counter)
            first = 1 if self.elitism and name == "Mutation" else 0

            members = p.map(func, self.population[first:])

            if self.elitism and name == "Mutation":
                members.append(self.population[0])

            self.population.current_population = members
            p.terminate()
        else:
            if self.progress_bar:
                pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], term_width=60, maxval=len(self.population)).start()
            ignor_first = self.elitism and name == "Mutation"

            for i, member in enumerate(self.population):
                if self.progress_bar:
                    pbar.update(i + 1)
                if not ignor_first:
                    member.apply_on_chromosome(current_function)
                ignor_first = False

        if self.progress_bar:
            pbar.finish()

        step_time = time.time() - start

        if step_time < 120:
            print('{0} time: {1:.2f}s\n'.format(name, step_time))
        else:
            print('{0} time: {1:.2f}min\n'.format(name, step_time // 60))

        return step_time, name

