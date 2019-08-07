from operator import attrgetter
import random
import numpy as np
from functools import partial

from abc import ABC, abstractmethod
from pathos.multiprocessing import Pool
from multiprocessing import Manager

from elements.chromosome import Chromosome, Particle


class PopulationBase(ABC):
    """Base class of population for metaheuristic algorithms."""

    __slots__ = "__current_generation", "pop_size", "chromosome_size", "counter",\
                "fitness_function", "global_best_individual", "pool", "pool_size"

    def __init__(self, pop_size, chromosome_size, fitness_function, pool, pool_size):
        """Initialise the Population."""

        self.__current_population = []
        self.pop_size = pop_size
        self.chromosome_size = chromosome_size
        self.counter = -1
        self.fitness_function = fitness_function
        self.global_best_individual = None

        self.pool = pool
        self.pool_size = pool_size

        self.create_initial_population()

    def __repr__(self):
        """Return initialised Population representation in human readable form."""
        return repr((self.pop_size, self.__current_population))

    def __next__(self):
        self.counter += 1
        if self.counter < len(self.__current_population):
            return self.__current_population[self.counter]
        else:
            self.counter = -1
            raise StopIteration()

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.__current_population[index]

    def __len__(self):
        return len(self.__current_population)

    @property
    def current_population(self):
        return self.__current_population

    @current_population.setter
    def current_population(self, chromosomes):
        self.__current_population = chromosomes

    @abstractmethod
    def create_initial_population(self):
        """Create members of the first population randomly."""
        pass

    def rank_population(self):
        """Sort the population by fitness ascending order."""
        self.__current_population.sort(key=attrgetter('fitness'), reverse=False)

    def add_individual_to_pop(self, individual):
        """Add an individual to the current population."""
        self.__current_population.append(individual)

    @abstractmethod
    def add_new_individual(self):
        """Add new individual to the current population."""
        pass

    def cut_pop_size(self):
        """Resize the current population to pop size."""
        self.__current_population = self.__current_population[:self.pop_size]

    def get_best_fitness(self):
        self.rank_population()
        return self.__current_population[0].fitness

    def get_best_genes(self):
        self.rank_population()
        return self.__current_population[0].genes

    def init_global_best(self):
        pass

    def set_global_best(self):
        """Set global best individual."""
        if self.__current_population[0].fitness < self.global_best_individual.fitness:
            self.global_best_individual.fitness = self.__current_population[0].fitness
            self.global_best_individual.genes = self.__current_population[0].genes.copy()

    @abstractmethod
    def init_global_and_personal_best(self):
        pass

    def get_global_best(self):
        """Get global best individual."""
        return self.global_best_individual

    def set_personal_bests(self):
        pass


class Population(PopulationBase):
    """ Population class that encapsulates all of the chromosomes."""

    def __init__(self, pop_size, chromosome_size, fitness_function, pool, pool_size):
        super().__init__(pop_size, chromosome_size, fitness_function, pool, pool_size)

        self.global_best_individual = Chromosome(self.chromosome_size, self.fitness_function)

    def create_initial_population(self):
        """Create members of the first population randomly."""

        for _ in range(self.pop_size):
            individual = Chromosome(self.chromosome_size, self.fitness_function)
            if not self.pool:
                individual.calculate_fitness()
            self.add_individual_to_pop(individual)

        if self.pool:
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)

            def pool_function(inside_lock, inside_counter, inside_member):
                inside_lock.acquire()
                inside_counter.value += 1
                inside_lock.release()

                fitness_value = inside_member.calculate_fitness(gpu=inside_counter.value % 4)

                return fitness_value

            func = partial(pool_function, lock, counter)
            fitness_values = p.map(func, self.current_population[:])

            for value, member in zip(fitness_values, self.current_population[:]):
                member.fitness = value

            p.terminate()

    def add_new_individual(self, *args, **kwargs):
        """Add new individual with fitness value to the current population."""

        individual = Chromosome(self.chromosome_size, self.fitness_function)
        individual.calculate_fitness(**kwargs)

        return individual

    def init_global_and_personal_best(self):
        self.global_best_individual.fitness = self.current_population[0].fitness
        self.global_best_individual.genes = self.current_population[0].genes.copy()

    def crossover(self, selection_function, *args, **kwargs):
        """Add new individuals to the population with crossover."""

        child_1 = Chromosome(self.chromosome_size, self.fitness_function)
        child_2 = Chromosome(self.chromosome_size, self.fitness_function)

        parent_1 = selection_function(self)
        parent_2 = selection_function(self)

        crossover_index = random.randrange(1, self.chromosome_size - 1)

        child_1.genes = np.concatenate((parent_1.genes[:crossover_index], parent_2.genes[crossover_index:]), axis=None)
        child_2.genes = np.concatenate((parent_2.genes[:crossover_index], parent_1.genes[crossover_index:]), axis=None)

        child_1.calculate_fitness(**kwargs)
        child_2.calculate_fitness(**kwargs)

        return [child_1, child_2]

    def differential_evolution(self, CR, F, current_index, **kwargs):
        """Modify the population via methods of differential evolution."""

        assert 0 <= CR <= 1
        assert 0 <= F <= 2

        all_indexes = range(len(self))

        # for current_index in all_indexes:
        valid_indexes = set(all_indexes) - {current_index}
        a_index, b_index, c_index = random.sample(valid_indexes, 3)

        a_genes = np.array(self[a_index].genes)
        b_genes = np.array(self[b_index].genes)
        c_genes = np.array(self[c_index].genes)

        member = self[current_index]
        donor_genes = a_genes + F * (b_genes - c_genes)
        member.set_test()
        random_index = random.choice(range(len(member)))

        for i in range(len(member)):
            if i == random_index or CR > np.random.rand():
                member.genes_test[i] = donor_genes[i]

        member.resize_invalid_genes_test()
        member.calculate_fitness_test(**kwargs)
        member.apply_test_if_better()

        return member

    def invasive_weed(self, iteration, iter_max, e, sigma_init, sigma_fin, N_min, N_max, member, **kwargs):
        """Add new individuals to the population via methods of invasive weed algorithm."""

        seeds = []
        sigma = ((iter_max - iteration) / iter_max)
        sigma = pow(sigma, e)
        sigma = sigma * (sigma_init - sigma_fin) + sigma_fin

        fitness_max = self.current_population[-1].fitness
        fitness_min = self.current_population[0].fitness

        ratio = (fitness_max - member.fitness) / (fitness_max - fitness_min)
        N = int(N_min + (N_max - N_min) * ratio)

        for _ in range(N):
            seed = Chromosome(self.chromosome_size, self.fitness_function)
            for i in range(self.chromosome_size):
                random_value = random.uniform(member.genes[i] - sigma, member.genes[i] + sigma)
                seed.genes[i] = random_value

            seed.resize_invalid_genes()
            seed.calculate_fitness(**kwargs)
            seeds.append(seed)

        return seeds


class Swarm(PopulationBase):
    """ Swarm class that encapsulates all of the particles."""

    def __init__(self, pop_size, chromosome_size, fitness_function, pool, pool_size):
        super().__init__(pop_size, chromosome_size, fitness_function, pool, pool_size)

        self.global_best_individual = Particle(self.chromosome_size, self.fitness_function)

    def create_initial_population(self):
        """Create members of the first population randomly."""

        for _ in range(self.pop_size):
            individual = Particle(self.chromosome_size, self.fitness_function)
            if not self.pool:
                individual.calculate_fitness()
            self.add_individual_to_pop(individual)

        if self.pool:
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)

            def pool_function(inside_lock, inside_counter, inside_member):
                inside_lock.acquire()
                inside_counter.value += 1
                inside_lock.release()

                fitness_value = inside_member.calculate_fitness(gpu=inside_counter.value % 4)

                return fitness_value

            func = partial(pool_function, lock, counter)
            fitness_values = p.map(func, self.current_population[:])

            for value, member in zip(fitness_values, self.current_population[:]):
                member.fitness = value

            p.terminate()

    def add_new_individual(self):
        """Add new individual with fitness value to the current population."""

        individual = Particle(self.chromosome_size, self.fitness_function)
        individual.calculate_fitness()
        self.add_individual_to_pop(individual)

    def init_global_and_personal_best(self):
        self.global_best_individual.fitness = self.current_population[0].fitness
        self.global_best_individual.genes = self.current_population[0].genes.copy()

        for particle in self.current_population:
            particle.personal_best = particle.genes
            particle.personal_best_fitness = particle.fitness

    def set_personal_bests(self):
        for particle in self.current_population:
            particle.set_personal_best()


