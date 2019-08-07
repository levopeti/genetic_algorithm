import numpy as np
import math
from abc import ABC, abstractmethod

from fitness_functions.fully_connected_nn import train_model_fc
from fitness_functions.conv_nn import train_model_cn


class FitnessFunctionBase(ABC):
    """Base class of fitness function for metaheuristic algorithms."""

    def __init__(self):
        self.name = None

    def calculate(self, genes, **kwargs):
        phenotype_of_genes = self.genotype_to_phenotype(genes=genes, **kwargs)
        return self.fitness_function(phenotype_of_genes)

    @abstractmethod
    def fitness_function(self, phenotype_of_genes):
        """Calculate fitness value from the genes."""
        pass

    @staticmethod
    @abstractmethod
    def genotype_to_phenotype(genes, **kwargs):
        """Transform genotype (genes) to phenotype for the fitness function."""
        pass


class RastriginFunction(FitnessFunctionBase):
    """https://en.wikipedia.org/wiki/Rastrigin_function"""

    def __init__(self):
        super().__init__()

        self.name = "rastrigin function"

    def fitness_function(self, phenotype_of_genes):
        n = len(phenotype_of_genes)
        return 10 * n + sum([i * i - 10 * np.cos(2 * i * np.pi) for i in phenotype_of_genes])

    def genotype_to_phenotype(self, genes, **kwargs):
        phenotype_of_genes = (np.array(genes) * 10.24) - 5.12
        return phenotype_of_genes


class FullyConnected(FitnessFunctionBase):
    """Train a fully connected neural network on mnist from the fully_connected_nn.py."""

    def __init__(self):
        super().__init__()

        self.name = "fully connected"

    def fitness_function(self, phenotype_of_genes):
        # max_num_of_params = 21620815, 6 hidden layers
        result, num_of_params = train_model_fc(phenotype_of_genes)
        val_acc_ratio = 100 - (result * 100)  # [1, 100]
        num_of_params_ratio = np.log10(num_of_params)  # [3.89, 7.34]
        print("Number of parameters: {} / {}\nResult: {} / {}".format(num_of_params, num_of_params_ratio, result, val_acc_ratio))
        print(val_acc_ratio + (num_of_params_ratio / 5), '\n')

        return val_acc_ratio + (num_of_params_ratio / 5)

    @staticmethod
    def genotype_to_phenotype(genes, **kwargs):
        # TODO: param min max

        genes = np.array(genes)
        input_dict = {"gpu": [kwargs.get("gpu", 0)]}

        num_of_hidden_layers = int(math.floor(genes[0] * 4.999))
        input_dict["num_of_hidden_layers"] = num_of_hidden_layers

        size_of_layers = np.floor((genes[1:5] * 1490) + 5).astype("int")
        input_dict["size_of_layers"] = size_of_layers

        dropouts = genes[5:10] * 0.9
        input_dict["dropouts"] = dropouts

        learning_rate = np.power(10, -4 * genes[10])
        input_dict["learning_rate"] = learning_rate

        return input_dict

    @staticmethod
    def log_result(phenotype_of_genes, fitness_value):
        num_of_params = train_model_fc(phenotype_of_genes, only_parameters=True)
        num_of_params_ratio = np.log10(num_of_params)
        val_acc_ratio = fitness_value - (num_of_params_ratio / 5)
        accuracy = 100 - val_acc_ratio

        return {"Accuracy: ": str(accuracy) + '%',
                "Accuracy ratio: ": val_acc_ratio,
                "Number of parameters: ": num_of_params,
                "Number of parameters ratio: ": num_of_params_ratio / 5}


class ConvNet(FitnessFunctionBase):
    """Train a convolutional neural network on fashion_mnist from the conv_nn.py."""

    def __init__(self):
        super().__init__()

        self.name = "convnet"

    def fitness_function(self, phenotype_of_genes):
        # max_num_of_params = 21620815, 6 hidden layers
        result, num_of_params = train_model_cn(phenotype_of_genes)
        val_acc_ratio = 100 - (result * 100)  # [1, 100]
        num_of_params_ratio = np.log10(num_of_params)  # [3.89, 7.34]
        print("Number of parameters: {} / {}\nResult: {} / {}".format(num_of_params, num_of_params_ratio, result, val_acc_ratio))
        print(val_acc_ratio + (num_of_params_ratio / 5), '\n')

        return val_acc_ratio + (num_of_params_ratio / 5)

    @staticmethod
    def genotype_to_phenotype(genes, **kwargs):
        # TODO: param min max

        genes = np.array(genes)
        input_dict = {"gpu": [kwargs.get("gpu", 0)]}

        # strings
        optimizers = ('adam', 'nadam', 'sgd', 'rmsprop')
        index = int(math.floor(genes[0] * 3.999))
        input_dict["optimizer"] = optimizers[index]

        # strings
        activation = ('relu', 'sigmoid', 'tanh')
        index = int(math.floor(genes[1] * 2.999))
        input_dict["activation"] = activation[index]

        # int [1, 3]
        num_block = int(math.floor(genes[2] * 2.999)) + 1
        input_dict["num_block"] = num_block

        # int [1, 6]
        num_conv = int(math.floor(genes[3] * 5.999)) + 1
        input_dict["num_conv"] = num_conv

        # int [1, 6]
        num_dense = int(math.floor(genes[4] * 5.999)) + 1
        input_dict["num_dense"] = num_dense

        # int [1, 501]
        size_dense = int(math.floor(genes[5] * 500.999)) + 1
        input_dict["size_dense"] = size_dense

        # int [1, 11]
        kernel_size = int(math.floor(genes[6] * 10.999)) + 1
        input_dict["kernel_size"] = kernel_size

        # int [1, 51]
        filters = int(math.floor(genes[7] * 50.999)) + 1
        input_dict["filters"] = filters

        # float [0.00001 - 0.1]
        weight_decay = np.power(10, (-4 * genes[8]) - 1)
        input_dict["weight_decay"] = weight_decay

        # float [0, 1]
        dropout = genes[9] * 0.9
        input_dict["dropout"] = dropout

        # float [0.0001 - 1]
        learning_rate = np.power(10, -4 * genes[10])
        input_dict["learning_rate"] = learning_rate

        return input_dict

    @staticmethod
    def log_result(phenotype_of_genes, fitness_value):
        num_of_params = train_model_cn(phenotype_of_genes, only_parameters=True)
        num_of_params_ratio = np.log10(num_of_params)
        val_acc_ratio = fitness_value - (num_of_params_ratio / 5)
        accuracy = 100 - val_acc_ratio

        return {"Accuracy: ": str(accuracy) + '%',
                "Accuracy ratio: ": val_acc_ratio,
                "Number of parameters: ": num_of_params,
                "Number of parameters ratio: ": num_of_params_ratio / 5}




