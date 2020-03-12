import numpy as np
import time

from abc import ABC, abstractmethod

np.random.seed(int(time.time()))


class ChromosomeBase(ABC):
    """Base class of chromosome for metaheuristic algorithms."""

    __slots__ = "chromosome_size", "__fitness", "__genes",\
                "__fitness_test", "__genes_test", "fitness_function",\
                "num_fitness_eval", "counter"

    def __init__(self, chromosome_size, fitness_function):
        self.chromosome_size = chromosome_size
        self.__fitness = None
        self.__genes = []

        self.__fitness_test = None
        self.__genes_test = None

        self.create_individual()
        self.fitness_function = fitness_function

        self.num_fitness_eval = 0
        self.counter = -1

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form."""
        return repr((self.__fitness, self.__genes))

    def __next__(self):
        self.counter += 1
        if self.counter < self.chromosome_size:
            return self.__genes[self.counter]
        else:
            self.counter = -1
            raise StopIteration()

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.__genes[index]

    def __len__(self):
        return self.chromosome_size

    @abstractmethod
    def create_individual(self):
        """Create a candidate solution representation."""
        pass

    def calculate_fitness(self, **kwargs):
        """Calculate the fitness value of the chromosome."""
        self.__fitness = self.fitness_function.calculate(self.__genes, **kwargs)
        self.num_fitness_eval += 1
        return self.__fitness

    @property
    def fitness(self):
        return self.__fitness

    @property
    def genes(self):
        return self.__genes

    @fitness.setter
    def fitness(self, value):
        if value >= 0:
            self.__fitness = value
        else:
            raise ValueError("Fitness value must be greater or equal than 0!")

    @genes.setter
    def genes(self, genes):
        if len(genes) != self.chromosome_size:
            raise ValueError("Length of genes is not valid!")
        self.__genes = genes
        self.resize_invalid_genes()

    @property
    def genes_test(self):
        return self.__genes_test

    @genes_test.setter
    def genes_test(self, genes):
        if len(genes) != self.chromosome_size:
            raise ValueError("Length of genes test is not valid!")
        self.__genes_test = genes
        self.resize_invalid_genes_test()

    def calculate_fitness_test(self, **kwargs):
        """Calculate the test fitness value of the test genes."""
        if self.genes_test is None:
            raise ValueError("Genes test is not set!")

        self.__fitness_test = self.fitness_function.calculate(self.__genes_test, **kwargs)
        self.num_fitness_eval += 1

    def resize_invalid_genes(self):
        """Resize invalid genes to valid."""
        pass

    def resize_invalid_genes_test(self):
        """Resize invalid genes test to valid."""
        pass

    def set_test(self):
        """Set test values to current values."""
        self.genes_test = self.__genes.copy()
        self.__fitness_test = self.__fitness

    def apply_test(self):
        """Set current values to test values and set test values to None."""

        if self.__genes_test is None or self.__fitness_test is None:
            raise ValueError("Test values should not be None.")

        self.genes = self.__genes_test
        self.__fitness = self.__fitness_test

        self.__genes_test = None
        self.__fitness_test = None

    def reject_test(self):
        """Set test values to None."""
        self.__genes_test = None
        self.__fitness_test = None

    def apply_test_if_better(self):
        """
        Apply test values and return True if they are better
        and set to None, else return False.
        """

        if self.__genes_test is None or self.__fitness_test is None:
            raise ValueError("Test values should not be None.")

        # if test is better
        if self.__fitness_test < self.__fitness:
            self.genes = self.__genes_test
            self.__fitness = self.__fitness_test

            self.__genes_test = None
            self.__fitness_test = None

            return True

        # if original is better
        else:
            self.__genes_test = None
            self.__fitness_test = None

            return False

    def apply_on_chromosome(self, func, **kwargs):
        """Apply function on the chromosome."""
        func(self, **kwargs)


class Chromosome(ChromosomeBase):
    """ Chromosome class that encapsulates an individual's fitness and solution representation."""

    def __init__(self, chromosome_size, fitness_function):
        super().__init__(chromosome_size, fitness_function)

    def create_individual(self):
        """Create a candidate solution representation."""
        self.genes = np.random.rand(self.chromosome_size)

    def resize_invalid_genes(self):
        """Resize invalid genes to valid."""

        for i in range(self.chromosome_size):
            if self.genes[i] > 1:
                self.genes[i] = 1
            elif self.genes[i] < 0:
                self.genes[i] = 0

    def resize_invalid_genes_test(self):
        """Resize invalid genes test to valid."""

        for i in range(self.chromosome_size):
            if self.genes_test[i] > 1:
                self.genes_test[i] = 1
            elif self.genes_test[i] < 0:
                self.genes_test[i] = 0


class Particle(ChromosomeBase):
    """
    Particle class that encapsulates an individual's fitness,
    solution and velocity representation.
    """
    __slots__ = "velocity", "personal_best", "personal_best_fitness",\
                "global_best"

    def __init__(self, chromosome_size, fitness_function):
        super().__init__(chromosome_size, fitness_function)

        self.velocity = None
        self.personal_best = None
        self.global_best = None
        self.personal_best_fitness = None

        self.init_velocity()

    def create_individual(self):
        """Create a candidate solution representation."""
        self.genes = np.random.rand(self.chromosome_size)
        self.personal_best = self.genes.copy

    def init_velocity(self):
        self.velocity = np.random.rand(self.chromosome_size)

    def resize_invalid_genes(self):
        """Resize invalid genes to valid."""

        for i in range(self.chromosome_size):
            if self.genes[i] > 1:
                self.genes[i] = 1
            elif self.genes[i] < 0:
                self.genes[i] = 0

    def set_personal_best(self):
        if self.fitness < self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best = self.genes.copy()






