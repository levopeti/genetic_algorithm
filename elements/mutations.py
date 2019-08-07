import random
import numpy as np
import time

np.random.seed(int(time.time()))


def mutation_functions(mutation_type, mutation_probability=None, num_of_clones=None, mutation_random_sequence=False, **kwargs):
    """
    If mutation_random_sequence is True, the sequence in the genes is random.
    Number of clones by bacterial mutation is num_if_clones.
    """

    def basic_mutation(member):
        """Mutation on all genes with mutation_probability."""

        random_index = list(range(member.chromosome_size))
        if mutation_random_sequence:
            random.shuffle(random_index)

        new_genes = []
        for i in random_index:
            gene = member[i]
            if np.random.rand() < mutation_probability:
                gene = np.random.rand()
            new_genes.append(gene)

        member.genes = new_genes
        member.calculate_fitness()

    def bacterial_mutation(member):
        """Bacterial mutation over all genes."""

        random_index = list(range(member.chromosome_size))
        if mutation_random_sequence:
            random.shuffle(random_index)

        for i in random_index:
            for _ in range(num_of_clones):
                member.set_test()
                member.genes_test[i] = np.random.rand()
                member.calculate_fitness_test()
                member.apply_test_if_better()

    if mutation_type == "basic":
        return basic_mutation

    elif mutation_type == "bacterial":
        return bacterial_mutation

    elif mutation_type is None:
        return None

    else:
        raise ValueError("{} mutation type is not valid!".format(mutation_type))

