import time
from functools import partial

from elements.population import Swarm
from algorithms.base_alg_class import BaseAlgorithmClass

from pathos.multiprocessing import Pool
from multiprocessing import cpu_count, Manager
from progressbar import ProgressBar, Bar, Percentage, ETA


class ParticleSwarm(BaseAlgorithmClass):
    """
    Particle Swarm class.
    This is the main class that controls the functionality of the Particle Swarm Algorithm.
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

    def create_population(self):
        """
        Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        if self.population is None:
            self.population = Swarm(self.population_size, self.chromosome_size, self.fitness_function, self.pool, self.pool_size)

    def init_steps(self):
        """Initialize the iteration steps."""
        self.iteration_steps = []

        if self.config["swarm_iteration"]:
            self.iteration_steps.append(partial(self.modify_one_by_one_function, "Swarm iteration"))

        if self.config["memetic"]:
            self.iteration_steps.append(partial(self.modify_one_by_one_function, "Local search"))

    def modify_one_by_one_function(self, name):
        """Apply a function (local search, mutation) to all chromosomes."""
        start = time.time()
        if self.progress_bar:
            print("{}:".format(name))

        if name == "Swarm iteration":
            current_function = partial(self.swarm_iteration_function, self.population.global_best_individual.genes)
            name = name[:15] + ' ' + self.config["iteration_type"]
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
            members = p.map(func, self.population[:])

            self.population.current_population = members
            p.terminate()
        else:
            if self.progress_bar:
                pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], term_width=60, maxval=len(self.population)).start()

            for i, member in enumerate(self.population):
                if self.progress_bar:
                    pbar.update(i + 1)
                member.apply_on_chromosome(current_function)

        if self.progress_bar:
            pbar.finish()

        step_time = time.time() - start
        print('{0} time: {1:.2f}s\n'.format(name, step_time))

        return step_time, name

