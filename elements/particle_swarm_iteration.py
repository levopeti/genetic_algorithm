import numpy as np
import time


def swarm_iteration_functions(iteration_type, inertia, phi_p, phi_g, norm, c_w, c_p, c_g, **kwargs):
    """Return the given iteration function."""

    assert 0 <= c_w <= c_p <= c_g

    def pso_iteration(global_best_genes, particle, **kwargs):
        """Update the velocity vectors and update the genes according to them."""
        np.random.seed(int(time.time()))

        def normalization(vector):
            vector = np.array(vector)
            current_norm = np.linalg.norm(vector)
            if current_norm:
                vector = (vector * norm) / current_norm

            return vector

        r_p = np.random.rand()
        r_g = np.random.rand()

        # calculate and normalize the direction vectors
        personal_best_direction = normalization(particle.personal_best - particle.genes)
        global_best_direction = normalization(global_best_genes - particle.genes)
        velocity = normalization(particle.velocity)

        # update the velocity vector
        velocity = inertia * velocity + phi_p * r_p * personal_best_direction + phi_g * r_g * global_best_direction

        # update the genes of the particle
        particle.genes = particle.genes + velocity
        particle.velocity = velocity

        particle.resize_invalid_genes()
        particle.calculate_fitness(**kwargs)

        return particle

    def sso_iteration(global_best_genes, particle, **kwargs):
        """
        Modify the genes according to a random number (p).
        If p is less than c_w we do not modify the gene.
        If p is greater than or equal to c_w and less than c_p
        we modify the gene to the right gene from the personal best.
        If p is greater than or equal to c_p and less than c_g
        we modify the gene to the right gene from the global best.
        if p is greater than or equal to c_g
        we modify the gene to a random one [0, 1].
        """
        np.random.seed(int(time.time()))

        for i in range(particle.chromosome_size):
            p = np.random.rand()

            if p < c_w:
                continue
            elif p < c_p:
                particle.genes[i] = particle.personal_best[i]
            elif p < c_g:
                particle.genes[i] = global_best_genes[i]
            else:
                particle.genes[i] = np.random.rand()

        particle.calculate_fitness(**kwargs)

        return particle

    if iteration_type == "pso":
        return pso_iteration

    elif iteration_type == "sso":
        return sso_iteration

    elif iteration_type is None:
        return None

    else:
        raise ValueError("{} iteration type is not valid!".format(iteration_type))



