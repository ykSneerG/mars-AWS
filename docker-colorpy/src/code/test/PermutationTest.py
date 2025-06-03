import numpy as np
# from scipy.optimize import minimize_scalar
# from skimage import color
import matplotlib.pyplot as plt

import sys
import os

# Füge das Wurzelverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

#from src.handlersPredict import GradientMixGenerator

class GradientMixGenerator:

    @staticmethod
    def generate_dcs(gradient, dimension):
        """
        Generate a list of mixtures for a given gradient and dimension using NumPy.

        :param gradient: A list or array of gradient points (values from 0 to 1).
        :param dimension: The dimension of the space (1D, 2D, 3D, etc.).
        :return: A 2D NumPy array of mixtures where each row represents a mixture.
        """
        
        # Ensure gradient is a NumPy array
        gradient = np.asarray(gradient)

        # Generate meshgrid for all dimensions
        grids = np.meshgrid(*[gradient] * dimension, indexing="ij")

        # Reshape into a 2D array where each row is a combination
        result = np.stack(grids, axis=-1).reshape(-1, dimension)

        return result

    @staticmethod
    def generate_dcs_tolist(gradient, dimension):
        return GradientMixGenerator.generate_dcs(gradient, dimension).tolist()
    
    @staticmethod
    def generate_dcs_multilin_tolist(gradient: np.ndarray, dimension: int):
        """
        Generate a list of mixtures for a given gradient and dimension using NumPy.
        For each dimension, create a sequence where that dimension varies across the gradient values
        while other dimensions are 0.
        
        e.g. gradient = [0, 50, 100] and diemnsion = 3
        result = [
            [0, 0, 0],
            [50, 0, 0],
            [100, 0, 0],
            [0, 0, 0],
            [0, 50, 0],
            [0, 100, 0],
            [0, 0, 0],
            [0, 0, 50],
            [0, 0, 100],
        ]   
        """   
        
        result = []
        gradient = gradient.tolist()  # Convert to list if it's a NumPy array

        # For each dimension
        for dim in range(dimension):
            base = np.zeros(dimension)  # Create a base array of zeros
            
            # For each gradient value
            for g in gradient:
                mix = base.copy()
                mix[dim] = g
                result.append(mix.tolist())  # Convert to list before appending

        return result
        


steps = 4
color = 4

dcs_gradient = np.linspace(0, 100, steps)
dcs = GradientMixGenerator.generate_dcs_tolist(dcs_gradient, color)

for row in dcs:
    print(row)
    
print (len(dcs))