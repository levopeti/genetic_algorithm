import random


def memetic_functions(memetic, number_of_steps, step_size, lamarck_random_sequence, **kwargs):
    """
    Local search for all chromosomes in a given steps.
    :param memetic: If True, local search is active.
    :param number_of_steps: Number of steps to the direction of the negative gradient.
    :param step_size: Size of step.
    :param lamarck_random_sequence: If True, the sequence in the genes is random.
    """

    def local_search(member):
        for _ in range(number_of_steps):
            random_index = list(range(member.chromosome_size))
            if lamarck_random_sequence:
                random.shuffle(random_index)

            for i in random_index:
                # take one step positive direction
                member.set_test()
                member.genes_test[i] += step_size
                member.resize_invalid_genes_test()
                member.calculate_fitness_test()
                test_is_better = member.apply_test_if_better()

                if not test_is_better:
                    # take one step negative direction, if positive is not better
                    member.set_test()
                    member.genes_test[i] -= step_size
                    member.resize_invalid_genes_test()
                    member.calculate_fitness_test()
                    member.apply_test_if_better()

    if memetic:
        return local_search

    else:
        return None


