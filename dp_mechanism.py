import numpy as np

def cal_sensitivity_pdp(clip, dataset_size, lr):

    return 2 * lr * clip / dataset_size

def cal_sensitivity_gdp(clip, dataset_size):

    return 2 * clip / dataset_size


def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)
